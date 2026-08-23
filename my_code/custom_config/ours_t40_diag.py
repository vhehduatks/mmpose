"""Task 40-A — zero-GPU diagnostic: gravity from sensor<->coarse alignment.

Recipe per frame (hmd12 = [q_R(3), q_L(3), dists(3), h_head, h_L, h_R]):
  1. d = h_head (q_head = 0 in the LHF head-local frame).
  2. Solve <q_L,g> = h_L - h_head, <q_R,g> = h_R - h_head, |g| = 1
     -> g_lhf (up), sign by "up points away from the hands midpoint".
  3. Kabsch coarse{Head=1,LHand=4,RHand=7} <-> {0, q_L, q_R} -> R_lhf->ec.
  4. g_ec = R @ g_lhf.
  5. Temporal smoothing (normalized moving mean) at windows {1,5,15,31}.

Reference (DIAGNOSTIC ONLY, never in a trained path): g_ref = R_c2w^T @ y
from the t27_feat_cache's stored c2w, fid-matched per session.

Outputs per smoothing window: bias (angle of the per-sequence Wahba
constant rotation), per-frame residual std after removing it (the fatal
component), degeneracy rate. Gate: smoothed std <=5 deg -> proceed.
"""
import glob
import os
import sys

import numpy as np

T37 = "/mnt/linux_hdd_a/t37_quest_gbh_cache"
T27 = "/mnt/dataset_vol/t27_feat_cache"
COLLIN_MIN = 5e-3      # |q_L x q_R| threshold (m^2)
WINDOWS = (1, 5, 15, 31)


def solve_g_local(qL, qR, bL, bR):
    """One frame. Returns (g_up(3), degenerate flag)."""
    n = np.cross(qL, qR)
    nn = np.linalg.norm(n)
    if nn < COLLIN_MIN:
        return None, True
    G = np.stack([qL, qR])                     # (2,3)
    g0, *_ = np.linalg.lstsq(G, np.array([bL, bR]), rcond=None)
    disc = 1.0 - g0 @ g0
    if disc < 0:
        return None, True
    t = np.sqrt(disc)
    n = n / nn
    cands = [g0 + t * n, g0 - t * n]
    # Sign prior: the LHF frame's local z points head -> hands-midpoint
    # (approximately DOWN), so up must have a negative local-z component.
    # (The hands-midpoint itself lies in the triangle plane, where the two
    # candidates are identical — it cannot discriminate.)
    g = min(cands, key=lambda c: c[2])
    return g / np.linalg.norm(g), False


def kabsch(A, B):
    """R minimizing |R @ B_i - A_i| over centered point sets (N,3)."""
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    H = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def smooth_dirs(g, w):
    if w <= 1:
        return g
    k = w // 2
    out = np.empty_like(g)
    for i in range(len(g)):
        seg = g[max(0, i - k):i + k + 1]
        m = seg.mean(0)
        out[i] = m / max(np.linalg.norm(m), 1e-9)
    return out


def wahba_bias(gh, gr):
    """Constant rotation aligning estimated dirs to reference dirs."""
    return kabsch(gr, gh)                      # maps gh -> gr


def ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def unit_tests():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g_true = rng.normal(size=3)
        g_true /= np.linalg.norm(g_true)
        qL = rng.normal(size=3)
        qR = rng.normal(size=3)
        n = np.cross(qL, qR)
        n = n / np.linalg.norm(n)
        mirror = g_true - 2 * (g_true @ n) * n   # the other exact solution
        g, bad = solve_g_local(qL, qR, qL @ g_true, qR @ g_true)
        assert not bad
        # (a) constraints hold exactly
        assert abs(qL @ g - qL @ g_true) < 1e-9
        assert abs(qR @ g - qR @ g_true) < 1e-9
        # (b) g is one of the two exact solutions
        assert min(ang(g, g_true), ang(g, mirror)) < 1e-4
        # (c) the sign prior picks correctly whenever it is decisive
        if g_true[2] < mirror[2] - 1e-9:
            assert ang(g, g_true) < 1e-4
    # collinear -> degenerate
    _, bad = solve_g_local(np.array([1.0, 0, 0]), np.array([2.0, 0, 0]),
                           0.1, 0.2)
    assert bad
    # Kabsch on random rigid motions
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        th = rng.uniform(0, np.pi)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        B = (R.T @ (A - A.mean(0)).T).T
        R2 = kabsch(A, B)
        assert np.abs(R2 - R).max() < 1e-8
    print("unit tests OK")


def main():
    unit_tests()
    rows = {w: [] for w in WINDOWS}            # (bias_deg, res_std, res_mean)
    degen_frames = total_frames = 0
    sign_ok = sign_tot = 0
    n_sess = 0
    for split in ("Val", "Train"):
        for f37 in sorted(glob.glob(f"{T37}/{split}/*.npz")):
            f27 = os.path.join(T27, split, os.path.basename(f37))
            if not os.path.exists(f27):
                continue
            z37 = np.load(f37)
            z27 = np.load(f27)
            fid37 = z37["fid"]
            fid27 = z27["fid"]
            common, i37, i27 = np.intersect1d(fid37, fid27,
                                              return_indices=True)
            if len(common) < 40:
                continue
            coarse = z37["coarse"][i37]
            hmd = z37["hmd12"][i37]
            c2w = z27["c2w"][i27]
            mask = z27["mask"][i27] > 0
            g_ref = np.einsum("nij,j->ni", c2w[:, :3, :3].transpose(0, 2, 1),
                              np.array([0.0, 1.0, 0.0]))
            g_est = np.zeros((len(common), 3))
            ok = np.zeros(len(common), bool)
            last = None
            for i in range(len(common)):
                qR, qL = hmd[i, 0:3], hmd[i, 3:6]
                hh, hL, hR = hmd[i, 9], hmd[i, 10], hmd[i, 11]
                g_l, bad = solve_g_local(qL, qR, hL - hh, hR - hh)
                total_frames += 1
                if bad:
                    degen_frames += 1
                    if last is None:
                        continue
                    g_l = last                 # temporal carry-over
                else:
                    last = g_l
                    sign_tot += 1
                    if abs(g_l[2]) > 0.15:     # sign-prior margin healthy
                        sign_ok += 1
                A = coarse[i, [1, 4, 7]]
                # Unity CSV coords are LEFT-handed; ego-cam is right-handed.
                # Bridge with the frozen M = diag(1,1,-1) before Procrustes
                # (dot products — the height solve — are M-invariant).
                M = np.array([1.0, 1.0, -1.0])
                B = np.stack([np.zeros(3), qL * M, qR * M])
                R = kabsch(A, B)
                g_est[i] = R @ (g_l * M)
                ok[i] = True
            sel = ok & mask & (np.linalg.norm(g_ref, axis=-1) > 0.5)
            if sel.sum() < 40:
                continue
            n_sess += 1
            for w in WINDOWS:
                gs = smooth_dirs(g_est, w)[sel]
                gs = gs / np.linalg.norm(gs, axis=-1, keepdims=True)
                gr = g_ref[sel]
                Rb = wahba_bias(gs, gr)
                bias_deg = np.degrees(np.arccos(
                    np.clip((np.trace(Rb) - 1) / 2, -1, 1)))
                res = ang(np.einsum("ij,nj->ni", Rb, gs), gr)
                rows[w].append((bias_deg, res.std(), res.mean()))
    print(f"sessions used: {n_sess}; degeneracy rate: "
          f"{degen_frames/max(total_frames,1)*100:.1f}% "
          f"({degen_frames}/{total_frames}); "
          f"sign-margin healthy (|g_z|>0.15): {sign_ok/max(sign_tot,1)*100:.1f}%")
    print(f"{'win':>4} {'bias med':>9} {'res-std med':>12} "
          f"{'res-std p90':>12} {'res-mean med':>13}")
    for w in WINDOWS:
        a = np.array(rows[w])
        print(f"{w:4d} {np.median(a[:,0]):9.2f} {np.median(a[:,1]):12.2f} "
              f"{np.percentile(a[:,1],90):12.2f} {np.median(a[:,2]):13.2f}")
    np.save(os.path.dirname(os.path.abspath(__file__)) + "/t40_diag_rows.npy",
            {w: np.array(rows[w]) for w in WINDOWS}, allow_pickle=True)


if __name__ == "__main__":
    main()
