"""Task 35 — GBH-cascade cache export (xR-EgoPose).

Reuses ours_t31_xr_export.py verbatim, repointed at the V3 both_from_ground
checkpoint (paper Table 2's 34.06 arm). Its val_pipeline carries
EnhanceHMDInfo(mode='both_from_ground'), so gt_instance_labels.hmd_info
arrives 12-dim (9 base + 3 GBH heights), all GT-derived — the exporter's
forward is HMD-generic and needs no change.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python \
        my_code/custom_config/ours_t35_xr_export.py Val
"""

import sys
from pathlib import Path

import ours_t31_xr_export as base

base.CFG = ("my_code/custom_config/"
            "HMD_xregopose_cascaded_both_from_ground_v3_full_config.py")
base.CKPT = ("/mnt/dataset_vol/xr_egodataset_best/"
             "HMD_xregopose_cascaded_both_from_ground_v3_full/"
             "best_xregopose_Full Body_All_mpjpe_epoch_8.pth")
base.OUT_ROOT = Path("/mnt/linux_hdd_a/t35_xr_gbh_cache")

if __name__ == "__main__":
    base.main(sys.argv[1])
