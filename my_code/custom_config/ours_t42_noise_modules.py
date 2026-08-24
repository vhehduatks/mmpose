"""Task 42 / Arm D1+D2 (IEEE VR 2027) — inference-time robustness sweeps.

KinectEgoposeNoisyDataset: the paper loader with Gaussian noise added to the raw
Quest positions BEFORE the 12-dim hmd12 is built (so LHF and GBH are perturbed
consistently, exactly as a noisy device would produce them), plus a floor-height
error (all Guardian heights shifted by -floor_offset_m, i.e. the assumed floor is
floor_offset_m too high). Deterministic per frame (seeded by frame path). Used
only in val/test pipelines with the frozen headline checkpoint — nothing is trained.

    sigma_head_m   : isotropic std on hmd_pos_{x,y,z}
    sigma_ctrl_m   : isotropic std on left/right_pos_{x,y,z}
    floor_offset_m : additive floor error (signed), applied to all *_pos_y
"""

import hashlib

import numpy as np

from mmpose.registry import DATASETS
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)

_POS = {"hmd": ("hmd_pos_x", "hmd_pos_y", "hmd_pos_z"),
        "left": ("left_pos_x", "left_pos_y", "left_pos_z"),
        "right": ("right_pos_x", "right_pos_y", "right_pos_z")}


@DATASETS.register_module()
class KinectEgoposeNoisyDataset(KinectEgoposeDataset):

    def __init__(self, *, sigma_head_m: float = 0.0, sigma_ctrl_m: float = 0.0,
                 floor_offset_m: float = 0.0, noise_seed: int = 0, **kwargs):
        self.sigma_head_m = float(sigma_head_m)
        self.sigma_ctrl_m = float(sigma_ctrl_m)
        self.floor_offset_m = float(floor_offset_m)
        self.noise_seed = int(noise_seed)
        super().__init__(**kwargs)

    def _parse_frame(self, json_path: str, csv_row: dict, *args, **kwargs):
        row = dict(csv_row)
        h = hashlib.md5(f"{self.noise_seed}|{json_path}".encode()).hexdigest()
        rng = np.random.default_rng(int(h[:16], 16))
        for name, keys in _POS.items():
            sig = self.sigma_head_m if name == "hmd" else self.sigma_ctrl_m
            try:
                p = np.array([float(row[k]) for k in keys], dtype=np.float64)
            except (KeyError, ValueError):
                continue
            if sig > 0:
                p = p + rng.normal(0.0, sig, size=3)
            p[1] -= self.floor_offset_m
            for k, v in zip(keys, p):
                row[k] = f"{v:.9f}"
        return super()._parse_frame(json_path, row, *args, **kwargs)
