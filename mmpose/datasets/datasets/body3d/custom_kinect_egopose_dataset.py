# Copyright (c) OpenMMLab. All rights reserved.
"""
Azure Kinect v4 EgoPose Dataset

Maps the 32-joint Azure Kinect skeleton to the 16-joint xRegopose format.
Uses real VR device positions from synced_data.csv for HMD info instead
of estimating from skeleton keypoints.

Data layout expected under data_root:
    {batch}/{session}/ego_dataset/annotations/frame_XXXXXX.json
    {batch}/{session}/ego_dataset/images/frame_XXXXXX.jpg
    {batch}/{session}/synced_data.csv
"""

from mmpose.registry import DATASETS
from mmengine.dataset import BaseDataset

import os
import json
import csv
import numpy as np
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.logging import print_log
from ..utils import parse_pose_metainfo
import logging

# Get the directory containing this file for relative path resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Kinect v4 (32 joints) → xRegopose (16 joints) mapping ──────────────
# Each entry is the Kinect joint name corresponding to the xRegopose index.
KINECT_TO_XREGOPOSE = [
    'SPINE_CHEST',       # 0: Spine2
    'HEAD',              # 1: Head
    'SHOULDER_LEFT',     # 2: LeftArm
    'ELBOW_LEFT',        # 3: LeftForeArm
    'WRIST_LEFT',        # 4: LeftHand  (fallback → HAND_LEFT)
    'SHOULDER_RIGHT',    # 5: RightArm
    'ELBOW_RIGHT',       # 6: RightForeArm
    'WRIST_RIGHT',       # 7: RightHand (fallback → HAND_RIGHT)
    'HIP_LEFT',          # 8: LeftUpLeg
    'KNEE_LEFT',         # 9: LeftLeg
    'ANKLE_LEFT',        # 10: LeftFoot
    'FOOT_LEFT',         # 11: LeftToeBase
    'HIP_RIGHT',         # 12: RightUpLeg
    'KNEE_RIGHT',        # 13: RightLeg
    'ANKLE_RIGHT',       # 14: RightFoot
    'FOOT_RIGHT',        # 15: RightToeBase
]

# Fallback joints for wrists when confidence is low
WRIST_FALLBACK = {
    'WRIST_LEFT': 'HAND_LEFT',
    'WRIST_RIGHT': 'HAND_RIGHT',
}


@DATASETS.register_module(name='KinectEgoposeDataset')
class KinectEgoposeDataset(BaseDataset):
    """Azure Kinect v4 dataset for egocentric 3D pose estimation.

    Parses per-frame JSON annotations (32-joint skeleton), maps to 16-joint
    xRegopose format, and computes HMD info from actual VR sensor data
    (synced_data.csv).

    Args:
        data_mode (str): Dataset mode. Default: 'topdown'.
        data_root (str): Root directory (e.g. .../Train or .../Test).
        pipeline (list): Processing pipeline.
        test_mode (bool): Whether in test mode.
        sample_interval (int): Sample every N-th frame.
    """

    MM_TO_M = 1000.0
    METAINFO: dict = dict(from_file=os.path.join(_CURRENT_DIR, 'egopose_info.py'))

    def __init__(self,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 sample_interval: int = 1):
        self.data_mode = data_mode
        self.data_root = data_root
        self.sample_interval = sample_interval

        super().__init__(
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    # ── Metainfo ────────────────────────────────────────────────────────
    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)
        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        metainfo_keys = [
            'flip_indices', 'skeleton_links',
            'upper_body_ids', 'lower_body_ids',
        ]
        for key in metainfo_keys:
            if key in self._metainfo and key not in data_info:
                data_info[key] = deepcopy(self._metainfo[key])
        return data_info

    # ── Main loading entry point ────────────────────────────────────────
    def load_data_list(self) -> List[dict]:
        """Scan session directories, parse annotations + CSV, return samples."""
        sessions = self._find_sessions(self.data_root)
        print_log(f'Found {len(sessions)} sessions under {self.data_root}',
                  logger='current', level=logging.INFO)

        data_list: List[dict] = []
        skipped_frames = 0

        for session_dir in sessions:
            session_name = os.path.basename(session_dir)
            action = self._extract_action(session_name)

            # Load CSV for this session
            csv_path = os.path.join(session_dir, 'synced_data.csv')
            csv_lookup = self._load_csv(csv_path)
            if csv_lookup is None:
                print_log(f'Skipping session {session_name}: '
                          f'cannot read synced_data.csv',
                          logger='current', level=logging.WARNING)
                continue

            # Scan annotations
            ann_dir = os.path.join(session_dir, 'ego_dataset', 'annotations')
            img_dir = os.path.join(session_dir, 'ego_dataset', 'images')
            if not os.path.isdir(ann_dir):
                continue

            ann_files = sorted([
                f for f in os.listdir(ann_dir)
                if f.endswith('.json') and f.startswith('frame_')
            ])

            for ann_file in ann_files:
                json_path = os.path.join(ann_dir, ann_file)

                # Parse frame id from filename: frame_000042.json → 42
                frame_id = int(ann_file.replace('frame_', '').replace('.json', ''))

                # Only use frames that have matching CSV rows
                if frame_id not in csv_lookup:
                    skipped_frames += 1
                    continue

                # Image path
                img_file = ann_file.replace('.json', '.jpg')
                img_path = os.path.join(img_dir, img_file)
                if not os.path.isfile(img_path):
                    skipped_frames += 1
                    continue

                # Parse annotation
                sample = self._parse_frame(
                    json_path, csv_lookup[frame_id], img_path, action)
                if sample is not None:
                    data_list.append(sample)

        # Apply sample interval
        if self.sample_interval > 1:
            data_list = data_list[::self.sample_interval]

        print_log(f'Loaded {len(data_list)} samples from Kinect dataset '
                  f'(skipped {skipped_frames} frames)',
                  logger='current', level=logging.INFO)
        return data_list

    # ── Session discovery ───────────────────────────────────────────────
    def _find_sessions(self, root: str) -> List[str]:
        """Recursively find session directories containing ego_dataset/ and
        synced_data.csv under *root*."""
        sessions: List[str] = []
        if not os.path.isdir(root):
            return sessions

        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue

            # Check if this is a session directory
            has_csv = os.path.isfile(
                os.path.join(entry_path, 'synced_data.csv'))
            has_ego = os.path.isdir(
                os.path.join(entry_path, 'ego_dataset', 'annotations'))

            if has_csv and has_ego:
                sessions.append(entry_path)
            else:
                # Recurse into batch directories
                sessions.extend(self._find_sessions(entry_path))

        return sessions

    # ── CSV loading ─────────────────────────────────────────────────────
    def _load_csv(self, csv_path: str) -> Optional[Dict[int, dict]]:
        """Load synced_data.csv → {frame_id: row_dict}."""
        if not os.path.isfile(csv_path):
            return None
        try:
            lookup: Dict[int, dict] = {}
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_id = int(row['frame'])
                    lookup[frame_id] = row
            return lookup
        except Exception as e:
            print_log(f'Error reading CSV {csv_path}: {e}',
                      logger='current', level=logging.WARNING)
            return None

    # ── Frame parsing ───────────────────────────────────────────────────
    def _parse_frame(self, json_path: str, csv_row: dict,
                     img_path: str, action: str) -> Optional[dict]:
        """Parse a single frame annotation + CSV row into a data_info dict.

        Returns None if the frame should be skipped (e.g. no bodies).
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        if data.get('num_bodies', 0) == 0:
            return None

        # Build name → joint lookup for 2D and 3D
        skel_2d = {j['name']: j for j in data['skeleton_2d']}
        skel_3d = {j['name']: j for j in data['skeleton_3d']}

        # Map 32 Kinect joints → 16 xRegopose joints
        p2d = np.zeros((16, 2), dtype=np.float32)
        p3d = np.zeros((16, 3), dtype=np.float32)
        vis = np.zeros(16, dtype=np.float32)

        for xr_idx, kinect_name in enumerate(KINECT_TO_XREGOPOSE):
            j2d = skel_2d.get(kinect_name)
            j3d = skel_3d.get(kinect_name)

            if j2d is None or j3d is None:
                # Joint not found in annotation – leave as zero, invisible
                continue

            conf_3d = j3d.get('confidence', 0)

            # HAND/WRIST fallback: if wrist confidence < 1, try HAND joint
            if kinect_name in WRIST_FALLBACK and conf_3d < 1:
                fb_name = WRIST_FALLBACK[kinect_name]
                fb_2d = skel_2d.get(fb_name)
                fb_3d = skel_3d.get(fb_name)
                if fb_2d is not None and fb_3d is not None:
                    j2d = fb_2d
                    j3d = fb_3d
                    conf_3d = j3d.get('confidence', 0)

            p2d[xr_idx] = [j2d['u'], j2d['v']]
            p3d[xr_idx] = [j3d['x'], j3d['y'], j3d['z']]
            vis[xr_idx] = 1.0 if conf_3d >= 2 else 0.0

        # Convert mm → meters
        p3d /= self.MM_TO_M

        # Compute bbox from visible 2D keypoints (with 20px padding)
        visible_mask = vis > 0
        if visible_mask.any():
            vis_pts = p2d[visible_mask]
            pad = 20.0
            x_min = max(0, vis_pts[:, 0].min() - pad)
            y_min = max(0, vis_pts[:, 1].min() - pad)
            x_max = vis_pts[:, 0].max() + pad
            y_max = vis_pts[:, 1].max() + pad
            bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        else:
            # No visible joints – use full image as bbox
            bbox = np.array([[0, 0, 1920, 1080]], dtype=np.float32)

        # Compute HMD info from CSV sensor data
        head = np.array([
            float(csv_row['hmd_pos_x']),
            float(csv_row['hmd_pos_y']),
            float(csv_row['hmd_pos_z']),
        ], dtype=np.float32)
        left_hand = np.array([
            float(csv_row['left_pos_x']),
            float(csv_row['left_pos_y']),
            float(csv_row['left_pos_z']),
        ], dtype=np.float32)
        right_hand = np.array([
            float(csv_row['right_pos_x']),
            float(csv_row['right_pos_y']),
            float(csv_row['right_pos_z']),
        ], dtype=np.float32)
        hmd_info = self._preprocess_hmd_data(head, right_hand, left_hand)

        return {
            'img_path': img_path,
            'keypoints': p2d.reshape(1, 16, 2),
            'keypoint3d': p3d.reshape(1, 16, 3),
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'hmd_info': hmd_info.reshape(1, 9),
            'keypoints_visible': vis.reshape(1, 16),
            'action': np.array([action]),
        }

    # ── HMD preprocessing (adapted from H5CachedEgoposeDataset) ────────
    def _preprocess_hmd_data(self, head: np.ndarray,
                             right_hand: np.ndarray,
                             left_hand: np.ndarray) -> np.ndarray:
        """Compute 9-dim HMD info from head and hand positions.

        Creates a local coordinate system from the three tracking points,
        then computes relative hand positions and distances.

        Args:
            head: (3,) HMD position in meters.
            right_hand: (3,) right controller position in meters.
            left_hand: (3,) left controller position in meters.

        Returns:
            (9,) array: [right_local(3), left_local(3),
                         hand_dist, right_dist, left_dist]
        """
        # Create local coordinate system
        midpoint = (right_hand + left_hand) / 2.0
        z_axis = midpoint - head
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        hand_vector = right_hand - left_hand
        x_axis = np.cross(z_axis, hand_vector)
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            x_axis = np.array([1.0, 0.0, 0.0])

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Transform to local coordinates
        right_local = np.dot(rotation_matrix.T, (right_hand - head))
        left_local = np.dot(rotation_matrix.T, (left_hand - head))

        # Compute distances
        hand_distance = np.linalg.norm(right_local - left_local)
        right_distance = np.linalg.norm(right_local)
        left_distance = np.linalg.norm(left_local)

        return np.concatenate([
            right_local, left_local,
            [hand_distance, right_distance, left_distance]
        ]).astype(np.float32)

    # ── Helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _extract_action(session_name: str) -> str:
        """Extract action label from session folder name.

        E.g. 'Dancing1_20260214_180124' → 'Dancing1'
             'Gaming-Archery_20260214_173836' → 'Gaming-Archery'
        """
        parts = session_name.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:-2])
        return session_name

    def filter_data(self) -> List[dict]:
        """Filter annotations. Default returns all data."""
        return self.data_list
