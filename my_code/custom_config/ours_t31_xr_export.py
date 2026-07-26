"""Task 31 — xR-EgoPose per-frame coords export from the frozen
HMD_xregopose_cascaded_no_hmd model (GATE 31-0: no extrinsics, no sensors,
head has no 26a-style q => coords-only transfer; the npz mirrors the t27
cache schema with dummy geometry so WindowData/T27Full run unchanged with
--coords-from-cache).

One npz per sequence ({subject_action}__{env}), frames sorted by h5 row
index (the caches are stored in sequence order — verified in the gate).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python \
        my_code/custom_config/ours_t31_xr_export.py Val
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmpose.apis import init_model

CFG = "my_code/custom_config/HMD_xregopose_cascaded_no_hmd_config.py"
CKPT = ("/mnt/linux_hdd_a/mmpose_work_dirs/HMD_xregopose_cascaded_no_hmd/"
        "best_xregopose_Full Body_All_mpjpe_epoch_8.pth")
OUT_ROOT = Path("/mnt/linux_hdd_a/t31_xr_cache")
H5 = {"Train": "/mnt/dataset_vol/h5cache/train_cache_with_images.h5",
      "Val": "/mnt/dataset_vol/h5cache/val_cache_with_images.h5",
      # official TestSet — export/eval ONCE, at the end, predefined
      "Test": "/mnt/dataset_vol/h5cache/test_cache_with_images.h5"}


def main(split):
    init_default_scope("mmpose")
    cfg = Config.fromfile(CFG)
    model = init_model(CFG, CKPT, device="cuda").eval()
    h = model.head

    lc = cfg.train_dataloader if split == "Train" else cfg.val_dataloader
    # Pin the cache file: the config's "val" is the official TESTSET
    # (test_cache_v2, 115k — the single-touch set, NOT for internal val),
    # and its train file is the V2 format whose row order is not guaranteed
    # to match the V1 file used for sequence grouping below. Always read
    # the exact file that img_paths grouping uses.
    lc["dataset"]["cache_file"] = H5[split]
    lc["dataset"]["pipeline"] = cfg.val_pipeline
    lc["sampler"]["shuffle"] = False
    lc["batch_size"] = 64
    lc["num_workers"] = 6
    lc["drop_last"] = False
    loader = Runner.build_dataloader(lc)

    rows = []
    with torch.no_grad():
        for batch in loader:
            proc = model.data_preprocessor(batch, False)
            ds = proc["data_samples"]
            feats = model.extract_feat(proc["inputs"])
            hm = h.forward(feats)
            z = h.encoder(hm.to(torch.float32))
            hmd = torch.cat([d.gt_instance_labels.hmd_info for d in ds]
                            ).to(torch.float32)
            zp = h._fuse(z, h.hmd_linear(hmd))
            coarse = h.pose_decoder(zp).reshape(-1, 16, 3)
            # mirrors loss(): refine(coarse, hm, backbone_feat, z, hmd_info)
            refined = h.refine(coarse, hm, feats[-1], z, hmd_info=hmd)
            gt = torch.cat([d.gt_instance_labels.keypoint3d for d in ds]
                           ).reshape(-1, 16, 3)
            for i, d in enumerate(ds):
                rows.append((int(d.metainfo["h5_img_idx"]),
                             coarse[i].float().cpu().numpy(),
                             refined[i].float().cpu().numpy(),
                             gt[i].float().cpu().numpy()))
            if len(rows) % 6400 < 64:
                print(f"[{split}] {len(rows)} frames", flush=True)

    with h5py.File(H5[split], "r") as f:
        paths = [p.decode() for p in f["img_paths"][:]]
    assert max(r[0] for r in rows) < len(paths), \
        "h5_img_idx exceeds grouping file — dataset/grouping cache mismatch"
    seqs = {}
    for idx, coarse, refined, gt in rows:
        key = paths[idx].rsplit("/rgba/", 1)[0]
        key = "__".join(key.split("/")[-3:-1])       # subject_action__env
        seqs.setdefault(key, []).append((idx, coarse, refined, gt))

    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    eye = np.eye(4, dtype=np.float32)
    for key, rs in seqs.items():
        rs.sort(key=lambda r: r[0])
        n = len(rs)
        cols = list(zip(*rs))
        np.savez(out_dir / f"{key}.npz",
                 fid=np.array(cols[0], np.int64),
                 sampled=np.zeros((n, 16, 1), np.float16),   # dummy (no q)
                 z=np.zeros((n, 1), np.float16),             # dummy
                 coarse=np.stack(cols[1]), refined=np.stack(cols[2]),
                 gt=np.stack(cols[3]),
                 c2w=np.repeat(eye[None], n, 0),             # no extrinsics
                 m2w=np.repeat(eye[None], n, 0),
                 mask=np.ones(n, np.float32),
                 sens=np.zeros((n, 2, 3), np.float32),
                 srot=np.zeros((n, 2, 3, 3), np.float32))
    print(f"[{split}] wrote {len(seqs)} sequences, {len(rows)} frames "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
