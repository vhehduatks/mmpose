"""Task 41-A — world-consistency grid probe for the constant camera<->HMD
transform T, per participant, zero GPU.

p_w(t) = m2w(t) @ (R p_ec(t) + tT).  Score candidates R by:
  (1) session-mean world torso verticality  <R_m R a(t), y>  — accumulated
      as a 3x3 moment matrix M, so score(R) = sum(R * M) (near-free);
  (2) contact-foot world drift: per candidate, closed-form least-squares tT
      over centered contact segments + Guardian floor (y=0) rows; score =
      RMS within-segment drift + RMS contact-y.
Two stages: verticality ranks the full grid (~9k), drift rescopes the top
TOPK; a fine local grid around the joint argmax refines.

Diagnostic-only reference: per-participant cam2middle rotation
(frame_export meta/transforms) — never enters any trained path.

Hands are NOT used anywhere (Task-40 finding: wrist<->grip articulates).
"""
import glob
import os
import sys

import numpy as np

T37 = "/mnt/linux_hdd_a/t37_quest_gbh_cache"
FE = "/mnt/dataset_vol/frame_export"
PELV_L, PELV_R, ROOT = 8, 12, 0
TOE_L, TOE_R = 11, 15
VEL_TH = 0.03           # m/frame contact velocity threshold (~16 fps)
TOPK = 400
UP = np.array([0.0, 1.0, 0.0])


# ---------------- SO(3) sampling ----------------
def so3_grid(n_axis=200, n_ang=48):
    i = np.arange(n_axis)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_axis)
    theta = np.pi * (1 + 5 ** 0.5) * i
    axes = np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)], -1)
    angs = np.linspace(0, np.pi, n_ang, endpoint=False) + np.pi / n_ang
    A, G = np.meshgrid(np.arange(n_axis), angs, indexing="ij")
    return rodrigues(axes[A.ravel()], G.ravel())


def rodrigues(axis, ang):
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    K = np.zeros((len(axis), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang), np.cos(ang)
    return (np.eye(3)[None] + s[:, None, None] * K
            + (1 - c)[:, None, None] * (K @ K))


def local_grid(R0, max_ang=0.2, n=500, rng=None):
    rng = rng or np.random.default_rng(1)
    ax = rng.normal(size=(n, 3))
    an = rng.uniform(0, max_ang, n)
    return rodrigues(ax, an) @ R0[None]


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ---------------- data ----------------
def load_participant(part, split):
    files = sorted(glob.glob(f"{T37}/{split}/{part}__*.npz"))
    out = []
    for f in files:
        z = np.load(f)
        m = z["mask"] > 0 if "mask" in z.files else np.ones(len(z["fid"]), bool)
        if m.sum() < 10:
            continue
        out.append((z["coarse"][m], z["m2w"][m]))
    return out


# ---------------- scoring ----------------
def verticality_moment(sessions):
    M = np.zeros((3, 3))
    n = 0
    for coarse, m2w in sessions:
        a = coarse[:, ROOT] - 0.5 * (coarse[:, PELV_L] + coarse[:, PELV_R])
        a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
        b = np.einsum("tji,j->ti", m2w[:, :3, :3], UP)   # R_m^T y
        M += b.T @ a
        n += len(a)
    return M / n


def drift_score(sessions, Rc):
    """Contact-foot drift + floor RMS for ONE candidate rotation (3,3)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    segs = []          # (q_seg (n,3), Rm_seg (n,3,3))
    n_contact = n_tot = 0
    for coarse, m2w in sessions:
        Rm, tm = m2w[:, :3, :3], m2w[:, :3, 3]
        for foot in (TOE_L, TOE_R):
            q = np.einsum("tij,tj->ti", Rm, coarse[:, foot] @ Rc.T) + tm
            low = q[:, 1] < np.einsum(
                "tij,tj->ti", Rm,
                coarse[:, TOE_L + TOE_R - foot] @ Rc.T)[:, 1] + tm[:, 1]
            vel = np.linalg.norm(np.diff(q, axis=0), axis=-1)
            still = np.concatenate([[False], vel < VEL_TH])
            contact = low & still
            n_contact += contact.sum()
            n_tot += len(contact)
            idx = np.where(contact)[0]
            if len(idx) == 0:
                continue
            brk = np.where(np.diff(idx) > 1)[0]
            for seg in np.split(idx, brk + 1):
                if len(seg) < 3:
                    continue
                segs.append((q[seg], Rm[seg]))
    if not segs:
        return None, 0.0
    # closed-form tT: minimize sum |q~ + Rm~ tT|^2 + |(q + Rm tT).y|^2
    for qs, Rs in segs:
        qc = qs - qs.mean(0)
        Rcnt = Rs - Rs.mean(0)
        A += np.einsum("tij,tik->jk", Rcnt, Rcnt)
        b -= np.einsum("tij,ti->j", Rcnt, qc)
        ry = Rs[:, 1, :]                      # y-row
        A += ry.T @ ry
        b -= ry.T @ qs[:, 1]
    tT = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    res = []
    for qs, Rs in segs:
        p = qs + np.einsum("tij,j->ti", Rs, tT)
        res.append(np.linalg.norm(p - p.mean(0), axis=-1))
        res.append(np.abs(p[:, 1]))
    rms = np.sqrt(np.mean(np.concatenate(res) ** 2))
    return rms, n_contact / max(n_tot, 1)


def probe_participant(sessions, grid):
    M = verticality_moment(sessions)
    v = np.einsum("cij,ij->c", grid, M)
    top = np.argsort(-v)[:TOPK]
    best = None
    for ci in top:
        rms, crate = drift_score(sessions, grid[ci])
        if rms is None:
            continue
        sc = v[ci] - 3.0 * rms          # fixed a-priori weighting
        if best is None or sc > best[0]:
            best = (sc, grid[ci], v[ci], rms, crate)
    if best is None:
        return None
    # fine local pass
    for Rf in local_grid(best[1]):
        rms, crate = drift_score(sessions, Rf)
        if rms is None:
            continue
        sc = np.sum(Rf * M) - 3.0 * rms
        if sc > best[0]:
            best = (sc, Rf, np.sum(Rf * M), rms, crate)
    # landscape sharpness on the coarse joint score (top-K only)
    joint = []
    for ci in top[:100]:
        rms, _ = drift_score(sessions, grid[ci])
        if rms is not None:
            joint.append(v[ci] - 3.0 * rms)
    joint = np.array(joint)
    sharp = (best[0] - np.median(joint)) / (joint.std() + 1e-9)
    return best, sharp


def head_rotation_range(sessions):
    angs = []
    for _, m2w in sessions:
        R = m2w[:, :3, :3]
        angs.append(rot_angle(R[0], R[len(R) // 2]))
        angs.append(rot_angle(R[0], R[-1]))
    return max(angs) if angs else 0.0


# ---------------- unit tests ----------------
def synth(rng, n=800, rotate=True):
    ax = rng.normal(size=3)
    Rt = rodrigues(ax[None], np.array([rng.uniform(0.3, 1.2)]))[0]
    tt = rng.normal(scale=0.1, size=3)
    yaw = np.cumsum(rng.normal(scale=0.05 if rotate else 0.0, size=n))
    pit = 0.3 * np.sin(np.linspace(0, 6, n)) if rotate else np.zeros(n)
    Rm = rodrigues(np.repeat([[0, 1, 0]], n, 0), yaw) @ \
        rodrigues(np.repeat([[1, 0, 0]], n, 0), pit)
    tm = np.stack([np.sin(np.linspace(0, 3, n)),
                   1.6 + 0.02 * np.sin(np.linspace(0, 9, n)),
                   np.cos(np.linspace(0, 3, n))], -1)
    coarse = np.zeros((n, 16, 3))
    up_w = UP
    # world-frame body: root above pelvis above toes; contact toes at y=0
    root_w = tm - np.array([0, 1.1, 0])
    pelv_w = root_w - 0.45 * up_w
    toeL_w = np.stack([tm[:, 0] - 0.1, np.zeros(n), tm[:, 2]], -1)
    toeR_w = np.stack([tm[:, 0] + 0.1, np.zeros(n) + 0.001, tm[:, 2]], -1)
    inv = lambda P: np.einsum(  # world -> ec via (m2w, Rt, tt)
        "ij,tj->ti", Rt.T,
        np.einsum("tji,tj->ti", Rm, P - tm) - tt)
    coarse[:, ROOT] = inv(root_w)
    coarse[:, PELV_L] = inv(pelv_w - [0.09, 0, 0])
    coarse[:, PELV_R] = inv(pelv_w + [0.09, 0, 0])
    coarse[:, TOE_L] = inv(toeL_w)
    coarse[:, TOE_R] = inv(toeR_w)
    coarse += rng.normal(scale=0.01, size=coarse.shape)
    m2w = np.repeat(np.eye(4)[None], n, 0)
    m2w[:, :3, :3] = Rm
    m2w[:, :3, 3] = tm
    return [(coarse, m2w)], Rt


def unit_tests(grid):
    rng = np.random.default_rng(0)
    ses, Rt = synth(rng, rotate=True)
    best, sharp = probe_participant(ses, grid)
    err = rot_angle(best[1], Rt)
    print(f"unit: recovery err {err:.1f} deg (grid+local), sharp {sharp:.1f}")
    assert err < 8.0, err
    ses0, _ = synth(rng, rotate=False)
    b0 = probe_participant(ses0, grid)
    sharp0 = b0[1] if b0 is not None else 0.0
    print(f"unit: flat-control sharpness {sharp0:.1f} (rotating case {sharp:.1f})")
    assert sharp0 < sharp, (sharp0, sharp)
    print("unit tests OK")


def main():
    grid = so3_grid()
    print(f"grid: {len(grid)} candidates")
    unit_tests(grid)
    parts = sorted({os.path.basename(f).split("__")[0]
                    for s in ("Train", "Val")
                    for f in glob.glob(f"{T37}/{s}/*.npz")})
    print(f"{'part':8s} {'split':5s} {'angErr':>7s} {'sharp':>6s} "
          f"{'driftRMS':>9s} {'contact%':>8s} {'headRot':>8s}")
    results = []
    for part in parts:
        split = "Val" if glob.glob(f"{T37}/Val/{part}__*.npz") else "Train"
        sessions = load_participant(part, split)
        if not sessions:
            continue
        ref_f = f"{FE}/{part}/meta/transforms/egocam_left_to_egocam_middle.npz"
        R_ref = np.load(ref_f)["rotations"] if os.path.exists(ref_f) else None
        out = probe_participant(sessions, grid)
        if out is None:
            print(f"{part:8s} {split:5s}  no-contact")
            continue
        (sc, R, vv, rms, crate), sharp = out
        err = rot_angle(R, R_ref) if R_ref is not None else float("nan")
        hr = head_rotation_range(sessions)
        results.append((part, err, sharp, rms, crate, hr, R))
        print(f"{part:8s} {split:5s} {err:7.1f} {sharp:6.1f} "
              f"{rms:9.3f} {crate*100:8.1f} {hr:8.1f}")
    a = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in results])
    print(f"\nSUMMARY over {len(results)} participants: "
          f"angErr median {np.nanmedian(a[:,0]):.1f} deg "
          f"(p90 {np.nanpercentile(a[:,0],90):.1f}); "
          f"sharp median {np.median(a[:,1]):.1f}; "
          f"contact median {np.median(a[:,3])*100:.1f}%; "
          f"headRot median {np.median(a[:,4]):.0f} deg")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "t41_probe_results.npy"),
            {r[0]: {"R": r[6], "angErr": r[1], "sharp": r[2],
                    "driftRMS": r[3], "contact": r[4], "headRot": r[5]}
             for r in results}, allow_pickle=True)


if __name__ == "__main__":
    main()
