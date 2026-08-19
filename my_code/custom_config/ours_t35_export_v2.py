"""Task 35 — CORRECTED cache export on the V2 image family (Gate A finding:
both cascades were trained on train_cache_v2 images; the V1 `*_with_images`
files carry different renderings, degrading the frozen models ~19 mm
out-of-domain). Train/Test re-exported from V2; the ValSet exists only as V1
(hard-linked from the earlier exports, disclosed).

    ours_t35_export_v2.py {nohmd|gbh} {Train|Test}
"""

import sys
from pathlib import Path

import ours_t31_xr_export as base

_WD = "/mnt/dataset_vol/work_dirs_260305/work_dirs"
MODELS = {
    "nohmd": (base.CFG, base.CKPT, "/mnt/linux_hdd_a/t35v2_nohmd_cache"),
    "gbh": (("my_code/custom_config/"
             "HMD_xregopose_cascaded_both_from_ground_v3_full_config.py"),
            ("/mnt/dataset_vol/xr_egodataset_best/"
             "HMD_xregopose_cascaded_both_from_ground_v3_full/"
             "best_xregopose_Full Body_All_mpjpe_epoch_8.pth"),
            "/mnt/linux_hdd_a/t35v2_gbh_cache"),
    # Task 36 ladder rungs (dumped run configs — guaranteed ckpt-matched)
    "lhfhead": (f"{_WD}/HMD_xregopose_cascaded_head_from_ground_v3_full/"
                "HMD_xregopose_cascaded_head_from_ground_v3_full_config.py",
                f"{_WD}/HMD_xregopose_cascaded_head_from_ground_v3_full/"
                "best_xregopose_Full Body_All_mpjpe_epoch_10.pth",
                "/mnt/linux_hdd_a/t36_lhfhead_cache"),
    "lhfhand": (f"{_WD}/HMD_xregopose_cascaded_hand_from_ground_v3_full/"
                "HMD_xregopose_cascaded_hand_from_ground_v3_full_config.py",
                f"{_WD}/HMD_xregopose_cascaded_hand_from_ground_v3_full/"
                "best_xregopose_Full Body_All_mpjpe_epoch_9.pth",
                "/mnt/linux_hdd_a/t36_lhfhand_cache"),
    # R1 LHF-only: retrained in 36-A (best ep10 = 40.79 vs historical 41.60)
    "lhf": ("my_code/custom_config/HMD_xregopose_cascaded_lhf_only_v3_full_config.py",
            "work_dirs/HMD_xregopose_cascaded_lhf_only_v3_full/"
            "best_xregopose_Full Body_All_mpjpe_epoch_10.pth",
            "/mnt/linux_hdd_a/t36_lhf_cache"),
}

if __name__ == "__main__":
    which, split = sys.argv[1], sys.argv[2]
    assert split in ("Train", "Test"), "Val exists only in the V1 family"
    base.CFG, base.CKPT, out = MODELS[which]
    base.OUT_ROOT = Path(out)
    base.H5 = {"Train": "/mnt/dataset_vol/h5cache/train_cache_v2.h5",
               "Test": "/mnt/dataset_vol/h5cache/test_cache_v2.h5"}
    base.main(split)
