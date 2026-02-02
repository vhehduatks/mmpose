# Copyright (c) OpenMMLab. All rights reserved.
"""
H5 Direct Mo2Cap2 Dataset

This dataset directly loads data from the pre-existing mo2cap2 HDF5 chunk files.
No additional preprocessing required - uses the original chunk files as-is.

Performance: ~530k samples load in ~3-5 seconds (vs ~5-10 minutes with JSON parsing)

HDF5 Chunk Structure (each chunk has 1000 samples):
    - Images: (1000, 3, 256, 256) - RGB images in CHW format
    - Annot2D: (1000, 15, 2) - 2D keypoints
    - Annot3D: (1000, 15, 3) - 3D keypoints (in mm)
    - Heatmaps: (1000, 15, 32, 32) - Heatmaps
    - ZoomImages: (1000, 3, 256, 256) - Zoomed images
    - ZoomHeatmaps: (1000, 15, 32, 32) - Zoomed heatmaps
"""

from mmpose.registry import DATASETS
from mmengine.dataset import BaseDataset

import os
import h5py
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import logging

from mmengine.logging import print_log
from ..utils import parse_pose_metainfo


@DATASETS.register_module(name='H5Mo2Cap2Dataset')
class H5Mo2Cap2Dataset(BaseDataset):
    """Mo2Cap2 Dataset with direct HDF5 chunk loading.

    This dataset directly reads from the original mo2cap2 HDF5 chunk files,
    providing extremely fast initialization (~3-5 seconds for 530k samples).

    The key advantage is that images are stored inside the H5 files,
    so no separate file I/O is needed during training.

    Args:
        data_mode (str): Dataset mode. Default: 'topdown'.
        metainfo (dict, optional): Meta information for the dataset.
        data_root (str, optional): Root directory containing chunk files.
        data_prefix (dict): Prefix for data path. Default: dict(img='').
        filter_cfg (dict, optional): Config for filtering data.
        indices (int or Sequence[int], optional): Indices to select.
        serialize_data (bool): Whether to serialize data. Default: True.
        pipeline (list): Processing pipeline. Default: [].
        test_mode (bool): Whether in test mode. Default: False.
        lazy_init (bool): Whether to use lazy init. Default: False.
        max_refetch (int): Max refetch times. Default: 1000.
        sample_interval (int): Sample interval. Default: 1.
        use_zoom (bool): Whether to use zoomed images. Default: False.
        chunk_pattern (str): Pattern for chunk files. Default: 'mo2cap2_chunk_*.hdf5'.
        preload_chunks (bool): Whether to keep H5 files open. Default: False.
    """

    MM_TO_M = 1000  # mo2cap2 uses millimeters
    NUM_KEYPOINTS = 15
    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_mo2cap2.py')

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
                 sample_interval: int = 1,
                 use_zoom: bool = False,
                 chunk_pattern: str = 'mo2cap2_chunk_',
                 preload_chunks: bool = False,
                 input_size: tuple = (256, 256)):
        """Initialize H5Mo2Cap2Dataset."""

        self.data_root = data_root
        self.data_mode = data_mode
        self.sample_interval = sample_interval
        self.use_zoom = use_zoom
        self.chunk_pattern = chunk_pattern
        self.preload_chunks = preload_chunks
        self.input_size = input_size

        # Build chunk index (very fast - just file listing)
        self._build_chunk_index()

        # Optionally preload chunk handles
        self._chunk_handles = {}

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

    def _build_chunk_index(self):
        """Build index of chunk files and sample mapping.

        This is very fast as it only lists files and reads their shapes.
        """
        print_log(f'Indexing chunk files from {self.data_root}...',
                  logger='current', level=logging.INFO)

        # Find all chunk files
        chunk_files = sorted([
            f for f in os.listdir(self.data_root)
            if f.startswith(self.chunk_pattern) and f.endswith('.hdf5')
        ])

        if len(chunk_files) == 0:
            raise ValueError(f'No chunk files found in {self.data_root} '
                           f'matching pattern {self.chunk_pattern}*.hdf5')

        self.chunk_files = [os.path.join(self.data_root, f) for f in chunk_files]
        self.chunk_sizes = []

        # Get size of each chunk (fast - just read attribute)
        total_samples = 0
        for chunk_path in self.chunk_files:
            with h5py.File(chunk_path, 'r') as hf:
                chunk_size = hf['Images'].shape[0]
                self.chunk_sizes.append(chunk_size)
                total_samples += chunk_size

        # Build cumulative index for fast lookup
        self.cumsum_sizes = np.cumsum([0] + self.chunk_sizes)
        self.total_samples = total_samples

        print_log(f'Found {len(self.chunk_files)} chunks, '
                  f'{total_samples} total samples',
                  logger='current', level=logging.INFO)

    def _get_chunk_and_local_idx(self, global_idx: int) -> tuple:
        """Convert global index to (chunk_idx, local_idx).

        Args:
            global_idx: Global sample index

        Returns:
            Tuple of (chunk_idx, local_idx)
        """
        chunk_idx = np.searchsorted(self.cumsum_sizes[1:], global_idx, side='right')
        local_idx = global_idx - self.cumsum_sizes[chunk_idx]
        return chunk_idx, local_idx

    def _get_chunk_handle(self, chunk_idx: int) -> h5py.File:
        """Get H5 file handle for a chunk.

        Args:
            chunk_idx: Index of chunk file

        Returns:
            h5py.File handle
        """
        if self.preload_chunks:
            if chunk_idx not in self._chunk_handles:
                self._chunk_handles[chunk_idx] = h5py.File(
                    self.chunk_files[chunk_idx], 'r')
            return self._chunk_handles[chunk_idx]
        else:
            return h5py.File(self.chunk_files[chunk_idx], 'r')

    def _preprocess_hmd_data_batch(self, p3d_batch: np.ndarray) -> np.ndarray:
        """Vectorized HMD data preprocessing for batch of 3D keypoints.

        Args:
            p3d_batch: (N, 15, 3) array of 3D keypoints

        Returns:
            (N, 9) array of HMD preprocessed features
        """
        N = p3d_batch.shape[0]

        # Mo2Cap2 skeleton: 0=Neck, 3=RightHand, 6=LeftHand
        head = p3d_batch[:, 0]        # (N, 3)
        right_hand = p3d_batch[:, 3]  # (N, 3)
        left_hand = p3d_batch[:, 6]   # (N, 3)

        # Create local coordinate system
        midpoint = (right_hand + left_hand) / 2  # (N, 3)
        z_axis = midpoint - head  # (N, 3)
        z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)  # (N, 1)
        z_norm = np.maximum(z_norm, 1e-6)  # Avoid division by zero
        z_axis = z_axis / z_norm  # (N, 3)

        hand_vector = right_hand - left_hand  # (N, 3)
        x_axis = np.cross(z_axis, hand_vector)  # (N, 3)
        x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)  # (N, 1)
        x_norm = np.maximum(x_norm, 1e-6)
        x_axis = x_axis / x_norm  # (N, 3)

        y_axis = np.cross(z_axis, x_axis)  # (N, 3)

        # Build rotation matrices: (N, 3, 3)
        rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

        # Transform to local coordinates using batch matrix multiplication
        right_rel = right_hand - head  # (N, 3)
        left_rel = left_hand - head  # (N, 3)

        # (N, 3, 3).T @ (N, 3, 1) -> einsum is cleaner
        right_local = np.einsum('nij,nj->ni', rotation_matrices, right_rel)  # (N, 3)
        left_local = np.einsum('nij,nj->ni', rotation_matrices, left_rel)  # (N, 3)

        # Compute distances
        hand_distance = np.linalg.norm(right_local - left_local, axis=1, keepdims=True)  # (N, 1)
        right_distance = np.linalg.norm(right_local, axis=1, keepdims=True)  # (N, 1)
        left_distance = np.linalg.norm(left_local, axis=1, keepdims=True)  # (N, 1)

        # Concatenate features: (N, 9)
        preprocessed_hmd = np.concatenate([
            right_local, left_local,
            hand_distance, right_distance, left_distance
        ], axis=1)

        return preprocessed_hmd.astype(np.float32)

    def _get_cache_path(self) -> str:
        """Get path to annotation cache file."""
        return os.path.join(self.data_root, 'annotations_cache.h5')

    def _load_or_build_cache(self):
        """Load annotations from cache or build if not exists."""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            print_log(f'Loading annotation cache from {cache_path}...',
                      logger='current', level=logging.INFO)
            with h5py.File(cache_path, 'r') as hf:
                all_keypoints = hf['keypoints'][:]
                all_keypoint3d = hf['keypoint3d'][:]
                all_hmd_info = hf['hmd_info'][:]
            print_log(f'Cache loaded: {all_keypoints.shape[0]} samples',
                      logger='current', level=logging.INFO)
            return all_keypoints, all_keypoint3d, all_hmd_info

        # Build cache
        print_log(f'Building annotation cache (this is one-time only)...',
                  logger='current', level=logging.INFO)

        all_keypoints = []
        all_keypoint3d = []
        all_hmd_info = []

        from tqdm import tqdm
        for chunk_idx, chunk_path in enumerate(tqdm(self.chunk_files, desc='Loading chunks')):
            with h5py.File(chunk_path, 'r') as hf:
                annot2d = hf['Annot2D'][:]  # (N, 15, 2)
                annot3d = hf['Annot3D'][:]  # (N, 15, 3)

                # Convert 3D to meters
                annot3d_m = annot3d / self.MM_TO_M

                # Subtract neck position (root-relative)
                annot3d_m = annot3d_m - annot3d_m[:, 0:1, :]

                # Vectorized HMD info computation
                chunk_hmd = self._preprocess_hmd_data_batch(annot3d_m)

                all_keypoints.append(annot2d)
                all_keypoint3d.append(annot3d_m)
                all_hmd_info.append(chunk_hmd)

        # Concatenate
        all_keypoints = np.concatenate(all_keypoints, axis=0).astype(np.float32)
        all_keypoint3d = np.concatenate(all_keypoint3d, axis=0).astype(np.float32)
        all_hmd_info = np.concatenate(all_hmd_info, axis=0).astype(np.float32)

        # Save cache
        print_log(f'Saving annotation cache to {cache_path}...',
                  logger='current', level=logging.INFO)
        with h5py.File(cache_path, 'w') as hf:
            hf.create_dataset('keypoints', data=all_keypoints, compression='gzip')
            hf.create_dataset('keypoint3d', data=all_keypoint3d, compression='gzip')
            hf.create_dataset('hmd_info', data=all_hmd_info, compression='gzip')
            hf.attrs['n_samples'] = len(all_keypoints)
            hf.attrs['version'] = '1.0'

        print_log(f'Cache saved: {len(all_keypoints)} samples',
                  logger='current', level=logging.INFO)
        return all_keypoints, all_keypoint3d, all_hmd_info

    def load_data_list(self) -> List[dict]:
        """Load data list by indexing all chunks.

        This creates lightweight data_info dicts that only store indices.
        Actual data loading happens in the pipeline via H5LoadImage transform.

        Returns:
            List of annotation dicts
        """
        data_list = []

        # Load or build annotation cache
        all_keypoints, all_keypoint3d, all_hmd_info = self._load_or_build_cache()

        # Apply sample interval
        indices = range(0, self.total_samples, self.sample_interval)
        print_log(f'Building data list for {len(list(indices))} samples...',
                  logger='current', level=logging.INFO)

        # Build data list
        bbox = np.array([[0, 0, self.input_size[0], self.input_size[1]]],
                       dtype=np.float32)

        for idx in indices:
            chunk_idx, local_idx = self._get_chunk_and_local_idx(idx)

            data_info = {
                # Store indices for lazy loading
                'h5_chunk_idx': chunk_idx,
                'h5_local_idx': local_idx,
                'h5_chunk_path': self.chunk_files[chunk_idx],
                'use_zoom': self.use_zoom,

                # Store pre-computed annotations
                'keypoints': all_keypoints[idx:idx+1],  # (1, 15, 2)
                'keypoint3d': all_keypoint3d[idx:idx+1],  # (1, 15, 3)
                'keypoints_visible': np.ones((1, self.NUM_KEYPOINTS), dtype=np.float32),
                'hmd_info': all_hmd_info[idx:idx+1],  # (1, 9)

                # Standard fields
                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'img_id': idx,
                'img_path': f'h5://{chunk_idx}/{local_idx}',  # Virtual path

                # raw_ann_info for metric compatibility (minimal placeholder)
                # CustomMo2Cap2Metric requires this field but only uses keypoint3d
                'raw_ann_info': {
                    'id': idx,
                    'image_id': idx,
                    'category_id': 1,
                    'keypoints': all_keypoints[idx].flatten().tolist() + [2] * self.NUM_KEYPOINTS,  # x,y,v format
                    'num_keypoints': self.NUM_KEYPOINTS,
                    'bbox': [0, 0, self.input_size[0], self.input_size[1]],
                    'area': self.input_size[0] * self.input_size[1],
                    'iscrowd': 0,
                },
            }
            data_list.append(data_info)

        print_log(f'Loaded {len(data_list)} samples from {len(self.chunk_files)} chunks',
                  logger='current', level=logging.INFO)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations. Default returns all data."""
        return self.data_list

    def __del__(self):
        """Clean up open file handles."""
        for handle in self._chunk_handles.values():
            handle.close()


@DATASETS.register_module(name='H5Mo2Cap2Dataset_Lazy')
class H5Mo2Cap2Dataset_Lazy(H5Mo2Cap2Dataset):
    """Lazy loading variant - loads keypoints only when needed.

    This variant has even faster initialization as it doesn't pre-load
    all keypoints. However, it may be slightly slower during training
    due to per-sample H5 reads.

    Use this if memory is constrained or for quick testing.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list with lazy keypoint loading.

        Only stores indices - actual data is loaded in pipeline.

        Returns:
            List of minimal annotation dicts
        """
        data_list = []

        # Apply sample interval
        indices = range(0, self.total_samples, self.sample_interval)

        print_log(f'Building lazy data list for {len(list(indices))} samples...',
                  logger='current', level=logging.INFO)

        bbox = np.array([[0, 0, self.input_size[0], self.input_size[1]]],
                       dtype=np.float32)

        for idx in indices:
            chunk_idx, local_idx = self._get_chunk_and_local_idx(idx)

            data_info = {
                'h5_chunk_idx': chunk_idx,
                'h5_local_idx': local_idx,
                'h5_chunk_path': self.chunk_files[chunk_idx],
                'use_zoom': self.use_zoom,
                'lazy_load': True,  # Flag for transform

                # Placeholder - will be loaded in transform
                'keypoints': None,
                'keypoint3d': None,
                'keypoints_visible': np.ones((1, self.NUM_KEYPOINTS), dtype=np.float32),
                'hmd_info': None,

                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'img_id': idx,
                'img_path': f'h5://{chunk_idx}/{local_idx}',
            }
            data_list.append(data_info)

        print_log(f'Created lazy data list with {len(data_list)} samples',
                  logger='current', level=logging.INFO)

        return data_list
