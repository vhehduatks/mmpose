"""Task 27-FULL — per-frame feature cache export from the frozen 26a model.

Stage-1 is frozen ⇒ soft-argmax sample locations are fixed per frame, so we
cache the per-joint SAMPLED backbone vectors (16,2048 fp16) instead of the
full 8x8x2048 map (~6-8 GB total vs ~45 GB). Also cached per frame: z (64
fp16), coarse (16,3 f32), live refined (16,3 f32; for the digit-for-digit
cache gate), GT keypoint3d, cam2world/mid2world/mask, sensors_v3 pos+rot.

One npz per session: {out_root}/{split}/{participant}__{session}.npz, frames
sorted by raw frame id (same per-session ordering the deployed STF2 cache
uses). Run in the mmpose env:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python \
        my_code/custom_config/ours_t27_cache.py Train
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

CFG = "my_code/custom_config/HMD_kinect_v5_t26_ecA_config.py"
CKPT = ("work_dirs/t26_ecA/"
        "best_xregopose_Full Body_All_mpjpe_epoch_6.pth")
OUT_ROOT = Path("/mnt/dataset_vol/t27_feat_cache")


def main(split, cfg_path=CFG, ckpt_path=CKPT, out_root=OUT_ROOT):
    out_root = Path(out_root)
    init_default_scope("mmpose")
    cfg = Config.fromfile(cfg_path)
    model = init_model(cfg_path, ckpt_path, device="cuda").eval()
    h = model.head

    lc = cfg.train_dataloader if split == "Train" else cfg.val_dataloader
    lc["dataset"]["pipeline"] = cfg.val_pipeline   # no train-time aug paths
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
            z = h.encoder(hm.to(torch.float32))
            coarse = h.pose_decoder(z).reshape(-1, 16, 3)   # _fuse == z (26a)
            sampled = h.sample_backbone(hm, feats[-1])      # (B,16,2048)
            h._prep(ds, hm.device)
            refined, _ = h._refine_core(coarse, sampled, z)
            labels = [d.gt_instance_labels for d in ds]
            gt = torch.cat([l.keypoint3d for l in labels]).reshape(-1, 16, 3)
            c2w = torch.cat([l.temporal_cam2world for l in labels])[:, 0]
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
                    sampled[i].half().cpu().numpy(),
                    z[i].half().cpu().numpy(),
                    coarse[i].float().cpu().numpy(),
                    refined[i].float().cpu().numpy(),
                    gt[i].float().cpu().numpy(),
                    c2w[i].float().cpu().numpy(),
                    m2w[i].float().cpu().numpy(),
                    float(mask[i]),
                    sens[i, :2].float().cpu().numpy(),
                    srot[i].float().cpu().numpy(),
                ))
                n += 1
            if n % 6400 < 32:
                print(f"[{split}] {n} frames", flush=True)

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for (part, sess), rows in acc.items():
        rows.sort(key=lambda r: r[0])
        cols = list(zip(*rows))
        np.savez(out_dir / f"{part}__{sess}.npz",
                 fid=np.array(cols[0], np.int64),
                 sampled=np.stack(cols[1]), z=np.stack(cols[2]),
                 coarse=np.stack(cols[3]), refined=np.stack(cols[4]),
                 gt=np.stack(cols[5]), c2w=np.stack(cols[6]),
                 m2w=np.stack(cols[7]), mask=np.array(cols[8], np.float32),
                 sens=np.stack(cols[9]), srot=np.stack(cols[10]))
    print(f"[{split}] wrote {len(acc)} sessions, {n} frames -> {out_dir}")


if __name__ == "__main__":
    # ours_t27_cache.py <split> [cfg] [ckpt] [out_root]
    main(*sys.argv[1:5])
