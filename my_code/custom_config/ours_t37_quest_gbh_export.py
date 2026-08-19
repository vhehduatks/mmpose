"""Task 37 — Quest 3 cache export from the frozen GBH headline cascade.

Model: the 64.36 mm `HMD_kinect_v5_flag_cascaded_ground_info_10ep` checkpoint
(LHF+GBH 12-dim, `CustomEgoposeCascadedRefinementHead_enhanced`), forwarded
manually in the proven t31 pattern (same head class as the xR GBH export).
Data: the t26 config's `KinectEgoposeSensorV3Dataset` dataloader — identical
image pipeline (EgoImageResize 256, codec 47x47 sigma 3) and
`ground_info_mode='both_from_ground'` (12-dim hmd_info from real CSV data),
plus per-frame calibration-free geometry labels.

CALIBRATION-FREE BY CONSTRUCTION (Task 37 directive): no ego-cam extrinsic is
written — `c2w` is identity filler for WindowData compat; the stored geometry
is m2w (HMD->world), sens/srot (controller world pose), hmd12.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python \
        my_code/custom_config/ours_t37_quest_gbh_export.py Val
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmpose.apis import init_model

MODEL_CFG = ("/mnt/dataset_vol/work_dir_260408/"
             "HMD_kinect_v5_flag_cascaded_ground_info_10ep/"
             "HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py")
MODEL_CKPT = ("/mnt/dataset_vol/work_dir_260408/"
              "HMD_kinect_v5_flag_cascaded_ground_info_10ep/"
              "best_xregopose_Full Body_All_mpjpe_epoch_10.pth")
DATA_CFG = "my_code/custom_config/HMD_kinect_v5_t26_ecA_config.py"
OUT_ROOT = Path("/mnt/linux_hdd_a/t37_quest_gbh_cache")


def main(split):
    init_default_scope("mmpose")
    model = init_model(MODEL_CFG, MODEL_CKPT, device="cuda").eval()
    h = model.head

    dcfg = Config.fromfile(DATA_CFG)
    lc = dcfg.train_dataloader if split == "Train" else dcfg.val_dataloader
    lc["dataset"]["pipeline"] = dcfg.val_pipeline
    lc["sampler"]["shuffle"] = False
    lc["batch_size"] = 32
    lc["num_workers"] = 6
    lc["drop_last"] = False
    loader = Runner.build_dataloader(lc)

    pat = re.compile(r"frame_(\d+)\.jpg$")
    acc = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            proc = model.data_preprocessor(batch, False)
            ds = proc["data_samples"]
            feats = model.extract_feat(proc["inputs"])
            hm = h.forward(feats)
            labels = [d.gt_instance_labels for d in ds]
            hmd = torch.cat([l.hmd_info for l in labels]).to(torch.float32)
            assert hmd.shape[-1] == 12, hmd.shape  # LHF 9 + GBH 3, real CSV
            # t31-pattern manual forward, mirrors decode()/loss()
            z = h.encoder(hm.to(torch.float32))
            zp = h._fuse(z, h.hmd_linear(hmd))
            coarse = h.pose_decoder(zp).reshape(-1, 16, 3)
            refined = h.refine(coarse, hm, feats[-1], z, hmd_info=hmd)
            gt = torch.cat([l.keypoint3d for l in labels]).reshape(-1, 16, 3)
            m2w = torch.cat([l.temporal_mid2world for l in labels])
            mask = torch.cat([l.temporal_mask for l in labels]).reshape(-1)
            sens = torch.cat([l.sensor_world for l in labels]).reshape(-1, 3, 3)
            srot = torch.cat([l.sensor_rot_world for l in labels]
                             ).reshape(-1, 2, 3, 3)
            for i, d in enumerate(ds):
                img = d.metainfo["img_path"]
                fid = int(pat.search(img).group(1))
                sess_dir = os.path.dirname(os.path.dirname(
                    os.path.dirname(img)))
                key = (os.path.basename(os.path.dirname(sess_dir)),
                       os.path.basename(sess_dir))
                acc.setdefault(key, []).append((
                    fid,
                    coarse[i].float().cpu().numpy(),
                    refined[i].float().cpu().numpy(),
                    gt[i].float().cpu().numpy(),
                    m2w[i].float().cpu().numpy(),
                    float(mask[i]),
                    sens[i, :2].float().cpu().numpy(),
                    srot[i].float().cpu().numpy(),
                    hmd[i].float().cpu().numpy(),
                ))
                n += 1
            if n % 6400 < 32:
                print(f"[{split}] {n} frames", flush=True)

    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    eye = np.eye(4, dtype=np.float32)
    for (part, sess), rows in acc.items():
        rows.sort(key=lambda r: r[0])
        nrow = len(rows)
        cols = list(zip(*rows))
        np.savez(out_dir / f"{part}__{sess}.npz",
                 fid=np.array(cols[0], np.int64),
                 sampled=np.zeros((nrow, 16, 1), np.float16),  # dummy (no q)
                 z=np.zeros((nrow, 1), np.float16),            # dummy
                 coarse=np.stack(cols[1]), refined=np.stack(cols[2]),
                 gt=np.stack(cols[3]),
                 c2w=np.repeat(eye[None], nrow, 0),  # NO T_ec_hmd: identity
                 m2w=np.stack(cols[4]),
                 mask=np.array(cols[5], np.float32),
                 sens=np.stack(cols[6]), srot=np.stack(cols[7]),
                 hmd12=np.stack(cols[8]))
    print(f"[{split}] wrote {len(acc)} sessions, {n} frames -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
