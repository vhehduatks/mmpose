"""Task 35-D — hmd12 side-channel for the temporal-stage oracle arm.

Derives the same 12-dim pseudo-HMD vector the V3 both_from_ground cascade
consumes (base 9-dim from the h5 `hmd_info` + 3 GBH heights recomputed from
GT `keypoint3d` with EnhanceHMDInfo's body-axis method, indices/math ported
verbatim from mmpose/datasets/transforms/enhance_hmd_info.py:102-152,217-232)
and writes one npz per sequence, `fid`-aligned to /mnt/linux_hdd_a/t31_xr_cache.

    /home/.conda/envs/mmpose/bin/python my_code/custom_config/ours_t35_hmd12.py Val
"""

import sys
from pathlib import Path

import h5py
import numpy as np

H5 = {"Train": "/mnt/dataset_vol/h5cache/train_cache_with_images.h5",
      "Val": "/mnt/dataset_vol/h5cache/val_cache_with_images.h5",
      "Test": "/mnt/dataset_vol/h5cache/test_cache_with_images.h5"}
T31 = Path("/mnt/linux_hdd_a/t31_xr_cache")
OUT = Path("/mnt/linux_hdd_a/t35_xr_hmd12")

ROOT, LHAND, RHAND = 0, 4, 7
LUPLEG, RUPLEG, LTOE, RTOE = 8, 12, 11, 15


def gbh(p3d):
    """(N,16,3) -> (N,3) [root_from_ground, lhand_fg, rhand_fg]; vectorized
    port of _compute_body_axis_ground + the both_from_ground branch."""
    root = p3d[:, ROOT]
    vec_a = (p3d[:, LUPLEG] + p3d[:, RUPLEG]) / 2 - root
    n = np.linalg.norm(vec_a, axis=-1, keepdims=True)
    unit = np.where(n > 1e-6, vec_a / np.maximum(n, 1e-12),
                    np.array([0, 0, 1], np.float32))
    lt = ((p3d[:, LTOE] - root) * unit).sum(-1)
    rt = ((p3d[:, RTOE] - root) * unit).sum(-1)
    ground = np.maximum(lt, rt)
    lh = ((p3d[:, LHAND] - root) * unit).sum(-1)
    rh = ((p3d[:, RHAND] - root) * unit).sum(-1)
    return np.stack([ground, ground - lh, ground - rh], -1).astype(np.float32)


def main(split):
    with h5py.File(H5[split], "r") as f:
        paths = [p.decode() for p in f["img_paths"][:]]
        hmd9 = f["hmd_info"][:].reshape(len(paths), 9).astype(np.float32)
        kp3d = f["keypoint3d"][:].reshape(len(paths), 16, 3).astype(np.float32)
    hmd12 = np.concatenate([hmd9, gbh(kp3d)], -1)  # (N, 12)

    out_dir = OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seq = 0
    for f31 in sorted((T31 / split).glob("*.npz")):
        fid = np.load(f31)["fid"]
        # sanity: fids index the same h5 the t31 cache was built from
        assert fid.max() < len(paths), f"{f31.name}: fid exceeds h5"
        key = paths[fid[0]].rsplit("/rgba/", 1)[0]
        key = "__".join(key.split("/")[-3:-1])
        assert key == f31.stem, f"{f31.name}: sequence identity mismatch ({key})"
        np.savez(out_dir / f31.name, fid=fid, hmd12=hmd12[fid])
        n_seq += 1
    print(f"[{split}] wrote {n_seq} sequences -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
