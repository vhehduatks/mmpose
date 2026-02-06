# Copyright (c) OpenMMLab. All rights reserved.
"""
Mo2Cap2 Test Dataset

Loads test data from:
- Images: JPG files in sequence directories (olek_outdoor, weipeng_studio)
- Ground Truth: MAT files with 'pose_gt' key (N, 15, 3)

Test data structure:
    /test_data/TestSet/
    ├── olek_outdoor/           # 2744 JPG images
    ├── olek_outdoor_gt.mat     # (3294, 15, 3) - Note: more GT than images
    ├── weipeng_studio/         # 2902 JPG images
    └── weipeng_studio_gt.mat   # GT poses
"""

from mmpose.registry import DATASETS
from mmengine.dataset import BaseDataset

import os
import numpy as np
import scipy.io as sio
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import logging

from mmengine.logging import print_log
from ..utils import parse_pose_metainfo


@DATASETS.register_module(name='Mo2Cap2TestDataset')
class Mo2Cap2TestDataset(BaseDataset):
    """Mo2Cap2 Test Dataset with JPG images and MAT ground truth.

    Args:
        data_root (str): Path to test_data/TestSet directory.
        sequence (str): Test sequence name ('olek_outdoor' or 'weipeng_studio').
        data_mode (str): Dataset mode. Default: 'topdown'.
        metainfo (dict, optional): Meta information for the dataset.
        pipeline (list): Processing pipeline. Default: [].
        test_mode (bool): Whether in test mode. Default: True.
        input_size (tuple): Input image size. Default: (256, 256).
    """

    MM_TO_M = 1000  # mo2cap2 uses millimeters
    NUM_KEYPOINTS = 15
    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_mo2cap2.py')

    def __init__(self,
                 data_root: str,
                 sequence: str = 'olek_outdoor',
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = True,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 input_size: tuple = (256, 256)):
        """Initialize Mo2Cap2TestDataset."""

        self.data_root = data_root
        self.sequence = sequence
        self.data_mode = data_mode
        self.input_size = input_size

        # Paths
        self.img_dir = os.path.join(data_root, sequence)
        self.gt_file = os.path.join(data_root, f'{sequence}_gt.mat')

        # Load ground truth and image list
        self._load_annotations()

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

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Load meta information."""
        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(f'metainfo should be a dict, but got {type(metainfo)}')

        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def _load_annotations(self):
        """Load image list and ground truth poses."""
        print_log(f'Loading test data from {self.sequence}...',
                  logger='current', level=logging.INFO)

        # Load image list (sorted by filename)
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f'Image directory not found: {self.img_dir}')

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.num_images = len(self.img_files)

        # Load ground truth
        if not os.path.exists(self.gt_file):
            raise FileNotFoundError(f'Ground truth file not found: {self.gt_file}')

        gt_data = sio.loadmat(self.gt_file)
        self.pose_gt = gt_data['pose_gt']  # (N, 15, 3) in millimeters

        print_log(f'Found {self.num_images} images, {self.pose_gt.shape[0]} GT poses',
                  logger='current', level=logging.INFO)

        # Note: GT may have more frames than images due to frame sync
        # We only use GT for the frames we have images
        if self.pose_gt.shape[0] < self.num_images:
            print_log(f'Warning: GT has fewer poses than images. '
                     f'Using {self.pose_gt.shape[0]} samples.',
                     logger='current', level=logging.WARNING)
            self.num_samples = self.pose_gt.shape[0]
        else:
            self.num_samples = self.num_images

        # Preprocess GT: convert to meters and make root-relative
        self.pose_gt_processed = self._preprocess_gt()

        # Compute HMD info for all samples
        self.hmd_info = self._compute_hmd_info_batch(self.pose_gt_processed)

    def _preprocess_gt(self) -> np.ndarray:
        """Preprocess ground truth poses.

        Returns:
            (N, 15, 3) poses in meters, root-relative (Neck at origin)
        """
        # Use only samples that have both images and GT
        pose_gt = self.pose_gt[:self.num_samples].copy()

        # Convert mm to meters
        pose_gt = pose_gt / self.MM_TO_M

        # Make root-relative (Neck = joint 0)
        pose_gt = pose_gt - pose_gt[:, 0:1, :]

        return pose_gt.astype(np.float32)

    def _compute_hmd_info_batch(self, p3d_batch: np.ndarray) -> np.ndarray:
        """Compute 9-dim HMD info for batch of poses.

        Args:
            p3d_batch: (N, 15, 3) root-relative poses

        Returns:
            (N, 9) HMD info
        """
        N = p3d_batch.shape[0]

        # Mo2Cap2 indices: 0=Neck, 3=RightHand, 6=LeftHand
        head = p3d_batch[:, 0]        # (N, 3)
        right_hand = p3d_batch[:, 3]  # (N, 3)
        left_hand = p3d_batch[:, 6]   # (N, 3)

        # Create local coordinate system
        midpoint = (right_hand + left_hand) / 2  # (N, 3)
        z_axis = midpoint - head  # (N, 3)
        z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)
        z_norm = np.maximum(z_norm, 1e-6)
        z_axis = z_axis / z_norm

        hand_vector = right_hand - left_hand
        x_axis = np.cross(z_axis, hand_vector)
        x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
        x_norm = np.maximum(x_norm, 1e-6)
        x_axis = x_axis / x_norm

        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrices: (N, 3, 3)
        rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

        # Transform to local coordinates
        right_rel = right_hand - head
        left_rel = left_hand - head

        right_local = np.einsum('nij,nj->ni', rotation_matrices, right_rel)
        left_local = np.einsum('nij,nj->ni', rotation_matrices, left_rel)

        # Compute distances
        hand_distance = np.linalg.norm(right_local - left_local, axis=1, keepdims=True)
        right_distance = np.linalg.norm(right_local, axis=1, keepdims=True)
        left_distance = np.linalg.norm(left_local, axis=1, keepdims=True)

        # Concatenate: (N, 9)
        hmd_info = np.concatenate([
            right_local, left_local,
            hand_distance, right_distance, left_distance
        ], axis=1)

        return hmd_info.astype(np.float32)

    def load_data_list(self) -> List[dict]:
        """Load data list.

        Returns:
            List of annotation dicts
        """
        data_list = []

        # Mo2Cap2 test images are 1280x1024, then center-cropped to 1024x1024
        # bbox should reflect the cropped image size for correct center/scale computation
        bbox = np.array([[0, 0, 1024, 1024]], dtype=np.float32)

        for idx in range(self.num_samples):
            img_path = os.path.join(self.img_dir, self.img_files[idx])

            data_info = {
                # Image path for LoadImage transform
                'img_path': img_path,

                # Annotations
                'keypoints': np.zeros((1, self.NUM_KEYPOINTS, 2), dtype=np.float32),  # 2D not used
                'keypoint3d': self.pose_gt_processed[idx:idx+1],  # (1, 15, 3)
                'keypoints_visible': np.ones((1, self.NUM_KEYPOINTS), dtype=np.float32),
                'hmd_info': self.hmd_info[idx:idx+1],  # (1, 9)

                # Standard fields
                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'img_id': idx,

                # Metadata for metric
                'raw_ann_info': {
                    'id': idx,
                    'image_id': idx,
                    'category_id': 1,
                    'keypoints': [0] * (self.NUM_KEYPOINTS * 3),
                    'num_keypoints': self.NUM_KEYPOINTS,
                    'bbox': [0, 0, self.input_size[0], self.input_size[1]],
                    'area': self.input_size[0] * self.input_size[1],
                    'iscrowd': 0,
                },

                # Action label (sequence name)
                'action': self.sequence,
            }
            data_list.append(data_info)

        print_log(f'Loaded {len(data_list)} test samples from {self.sequence}',
                  logger='current', level=logging.INFO)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations. Default returns all data."""
        return self.data_list


@DATASETS.register_module(name='Mo2Cap2CombinedTestDataset')
class Mo2Cap2CombinedTestDataset(BaseDataset):
    """Combined Mo2Cap2 Test Dataset (both sequences).

    Combines olek_outdoor and weipeng_studio for full test evaluation.
    """

    MM_TO_M = 1000
    NUM_KEYPOINTS = 15
    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_mo2cap2.py')

    def __init__(self,
                 data_root: str,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = True,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 input_size: tuple = (256, 256)):
        """Initialize combined dataset."""

        self.data_root = data_root
        self.data_mode = data_mode
        self.input_size = input_size
        self.sequences = ['olek_outdoor', 'weipeng_studio']

        # Load all sequences
        self._load_all_sequences()

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

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Load meta information."""
        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def _load_all_sequences(self):
        """Load all test sequences."""
        self.all_samples = []

        for seq in self.sequences:
            img_dir = os.path.join(self.data_root, seq)
            gt_file = os.path.join(self.data_root, f'{seq}_gt.mat')

            if not os.path.exists(img_dir) or not os.path.exists(gt_file):
                print_log(f'Skipping {seq}: files not found',
                         logger='current', level=logging.WARNING)
                continue

            # Load images and GT
            img_files = sorted([f for f in os.listdir(img_dir)
                              if f.endswith(('.jpg', '.png'))])
            gt_data = sio.loadmat(gt_file)
            pose_gt = gt_data['pose_gt']

            num_samples = min(len(img_files), pose_gt.shape[0])

            # Preprocess GT
            pose_gt = pose_gt[:num_samples] / self.MM_TO_M
            pose_gt = pose_gt - pose_gt[:, 0:1, :]

            # Compute HMD info
            hmd_info = self._compute_hmd_info_batch(pose_gt)

            for idx in range(num_samples):
                self.all_samples.append({
                    'img_path': os.path.join(img_dir, img_files[idx]),
                    'keypoint3d': pose_gt[idx].astype(np.float32),
                    'hmd_info': hmd_info[idx].astype(np.float32),
                    'action': seq,
                    'frame_idx': idx,  # 0-indexed frame within sequence
                    'sequence_name': seq,  # For official action lookup
                })

            print_log(f'Loaded {num_samples} samples from {seq}',
                     logger='current', level=logging.INFO)

        print_log(f'Total: {len(self.all_samples)} test samples',
                 logger='current', level=logging.INFO)

    def _compute_hmd_info_batch(self, p3d_batch: np.ndarray) -> np.ndarray:
        """Compute HMD info (same as Mo2Cap2TestDataset)."""
        N = p3d_batch.shape[0]
        head = p3d_batch[:, 0]
        right_hand = p3d_batch[:, 3]
        left_hand = p3d_batch[:, 6]

        midpoint = (right_hand + left_hand) / 2
        z_axis = midpoint - head
        z_norm = np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1e-6)
        z_axis = z_axis / z_norm

        hand_vector = right_hand - left_hand
        x_axis = np.cross(z_axis, hand_vector)
        x_norm = np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-6)
        x_axis = x_axis / x_norm

        y_axis = np.cross(z_axis, x_axis)
        rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

        right_rel = right_hand - head
        left_rel = left_hand - head
        right_local = np.einsum('nij,nj->ni', rotation_matrices, right_rel)
        left_local = np.einsum('nij,nj->ni', rotation_matrices, left_rel)

        hand_distance = np.linalg.norm(right_local - left_local, axis=1, keepdims=True)
        right_distance = np.linalg.norm(right_local, axis=1, keepdims=True)
        left_distance = np.linalg.norm(left_local, axis=1, keepdims=True)

        return np.concatenate([right_local, left_local,
                              hand_distance, right_distance, left_distance], axis=1)

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        data_list = []
        # Mo2Cap2 test images are 1280x1024, then center-cropped to 1024x1024
        # bbox should reflect the cropped image size for correct center/scale computation
        # in GetBBoxCenterScale and TopdownAffine transforms
        bbox = np.array([[0, 0, 1024, 1024]], dtype=np.float32)

        for idx, sample in enumerate(self.all_samples):
            data_info = {
                'img_path': sample['img_path'],
                'keypoints': np.zeros((1, self.NUM_KEYPOINTS, 2), dtype=np.float32),
                'keypoint3d': sample['keypoint3d'][np.newaxis, :],
                'keypoints_visible': np.ones((1, self.NUM_KEYPOINTS), dtype=np.float32),
                'hmd_info': sample['hmd_info'][np.newaxis, :],
                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'img_id': idx,
                'raw_ann_info': {
                    'id': idx, 'image_id': idx, 'category_id': 1,
                    'keypoints': [0] * (self.NUM_KEYPOINTS * 3),
                    'num_keypoints': self.NUM_KEYPOINTS,
                    'bbox': [0, 0, self.input_size[0], self.input_size[1]],
                    'area': self.input_size[0] * self.input_size[1],
                    'iscrowd': 0,
                },
                'action': sample['action'],
                'frame_idx': sample['frame_idx'],  # For official per-action evaluation
                'sequence_name': sample['sequence_name'],  # For official per-action evaluation
            }
            data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        return self.data_list
