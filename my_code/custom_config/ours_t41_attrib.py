"""Task 41-B attribution: per-participant Val MPJPE under three canon sources.

A = learned T_learn (deployed run, as-is)
B = grid-init T (the 41-A argmax, no learning)  [buffer wcalib_rot6_init]
C = oracle rig rotation from frame_export cam2middle  [DIAGNOSTIC ONLY]

Also: converged T_learn vs oracle rotation per TRAIN participant
(pre-registration #4).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/hyeonghwan/github/mmpose")

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmpose.apis import init_model

init_default_scope("mmpose")

CFG = "my_code/custom_config/HMD_kinect_v5_gbh_wcalib_config.py"
CKPT = sys.argv[1]
FE = "/mnt/dataset_vol/frame_export"

model = init_model(CFG, CKPT, device="cuda").eval()
h = model.head
from my_code.custom_config.ours_t_modules import compute_relpose_to_floor

# oracle rotations (diagnostic-only)
R_oracle = {}
for p in h._wc_part_index:
    f = f"{FE}/{p}/meta/transforms/egocam_left_to_egocam_middle.npz"
    if os.path.exists(f):
        R_oracle[p] = torch.from_numpy(
            np.load(f)["rotations"].reshape(3, 3)).float()

# T_learn vs oracle, per participant
print("== converged T_learn vs oracle rig rotation (deg) ==")
with torch.no_grad():
    Rl = h._rot6_to_R(h.wcalib_rot6.cpu())
    R0 = h._rot6_to_R(h.wcalib_rot6_init.cpu())
rows = []
for p, i in sorted(h._wc_part_index.items()):
    if p not in R_oracle:
        continue
    Ro = R_oracle[p]
    def ang(A, B):
        c = ((A * B).sum() - 1) / 2
        return float(torch.rad2deg(torch.arccos(c.clamp(-1, 1))))
    rows.append((p, ang(Rl[i], Ro), ang(R0[i], Ro), ang(Rl[i], R0[i])))
print(f"{'part':8s} {'learned-vs-oracle':>17s} {'init-vs-oracle':>14s} "
      f"{'drift':>6s}")
for p, a, b, d in rows:
    print(f"{p:8s} {a:17.1f} {b:14.1f} {d:6.1f}")
a = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"median: learned-vs-oracle {np.median(a[:,0]):.1f}, "
      f"init-vs-oracle {np.median(a[:,1]):.1f}, drift {np.median(a[:,2]):.1f}")

cfg = Config.fromfile(CFG)
lc = cfg.val_dataloader
lc["batch_size"] = 32
lc["num_workers"] = 6
loader = Runner.build_dataloader(lc)

err = {v: {} for v in "ABC"}          # part -> [sum_mm, n]
with torch.no_grad():
    for bi, batch in enumerate(loader):
        proc = model.data_preprocessor(batch, False)
        ds = proc["data_samples"]
        feats = model.extract_feat(proc["inputs"])
        hm = h.forward(feats)
        labels = [d.gt_instance_labels for d in ds]
        hmd = torch.cat([l.hmd_info for l in labels]).float()
        gt = torch.cat([l.keypoint3d for l in labels]).reshape(-1, 16, 3).cuda()
        m2w = torch.cat([l.temporal_mid2world for l in labels]).float().cuda()
        mask = torch.cat([l.temporal_mask for l in labels]).float().reshape(-1).cuda()
        parts = [h._part_of(d) for d in ds]
        z = h.encoder(hm.float())
        zp = h._fuse(z, h.hmd_linear(hmd.cuda()))
        coarse = h.pose_decoder(zp).reshape(-1, 16, 3)
        m2f = compute_relpose_to_floor(m2w)
        eye = torch.eye(4, device="cuda").expand_as(m2f)
        idx = torch.tensor([h._wc_part_index[p] for p in parts], device="cuda")
        variants = {}
        Rl_g = h._rot6_to_R(h.wcalib_rot6)[idx]
        R0_g = h._rot6_to_R(h.wcalib_rot6_init)[idx]
        Rc_g = torch.stack([R_oracle[p] for p in parts]).cuda()
        for v, R in (("A", Rl_g), ("B", R0_g), ("C", Rc_g)):
            T = torch.zeros(len(idx), 4, 4, device="cuda")
            T[:, :3, :3] = R
            T[:, 3, 3] = 1.0
            if v == "A":
                T[:, :3, 3] = h.wcalib_t[idx]
            canon = m2f @ T
            canon = torch.where(mask.reshape(-1, 1, 1) > 0, canon, eye)
            h._canon = canon
            h._wc_mask = mask
            refined = h.refine(coarse, hm, feats[-1], z, hmd_info=hmd.cuda())
            e = (refined - gt).norm(dim=-1).mean(-1) * 1000.0   # (B,) mm
            for k, p in enumerate(parts):
                s = err[v].setdefault(p, [0.0, 0])
                s[0] += float(e[k])
                s[1] += 1
            h._canon = None
            h._wc_mask = None
        if bi % 100 == 0:
            print(f"batch {bi}", flush=True)

print("\n== per-participant Val MPJPE (mm) ==")
print(f"{'part':8s} {'A=learned':>9s} {'B=grid':>8s} {'C=oracle':>9s}")
allv = {v: [] for v in "ABC"}
for p in sorted(err["A"]):
    line = [p]
    for v in "ABC":
        s = err[v][p]
        m = s[0] / s[1]
        allv[v].append((m, s[1]))
        line.append(m)
    print(f"{line[0]:8s} {line[1]:9.2f} {line[2]:8.2f} {line[3]:9.2f}")
for v in "ABC":
    tot = sum(m * n for m, n in allv[v]) / sum(n for _, n in allv[v])
    print(f"overall {v}: {tot:.2f} mm")
