"""Task 35 — hmd12 side-channel regenerated from the V2 h5 family (user
directive 2026-08-19: no run reads *_cache_with_images.h5). hmd_info and
keypoint3d are byte-identical between families, so this is a provenance
regeneration; fid alignment is against the corrected t35v2_nohmd_cache.
V2 has Train/Test only.

    ours_t35_hmd12_v2.py {Train|Test}
"""

import sys
from pathlib import Path

import ours_t35_hmd12 as base

base.H5 = {"Train": "/mnt/dataset_vol/h5cache/train_cache_v2.h5",
           "Test": "/mnt/dataset_vol/h5cache/test_cache_v2.h5"}
base.T31 = Path("/mnt/linux_hdd_a/t35v2_nohmd_cache")
base.OUT = Path("/mnt/linux_hdd_a/t35v2_hmd12")

if __name__ == "__main__":
    split = sys.argv[1]
    assert split in ("Train", "Test")
    base.main(split)
