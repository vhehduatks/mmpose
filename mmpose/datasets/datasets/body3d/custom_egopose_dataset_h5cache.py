# Copyright (c) OpenMMLab. All rights reserved.
"""
H5 Cached EgoPose Dataset

This dataset implementation pre-processes all annotations into a single HDF5 file
for fast loading during training. Instead of parsing 65k+ JSON files at init time,
it loads pre-computed numpy arrays from a single H5 file.

Performance improvement: ~10 minutes -> ~5 seconds loading time
"""

from mmpose.registry import DATASETS
from mmengine.dataset import BaseDataset

import os
import json
import h5py
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from tqdm import tqdm
import logging

from mmengine.logging import print_log
from .config import config
from ..utils import parse_pose_metainfo

# Get the directory containing this file for relative path resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@DATASETS.register_module(name='H5CachedEgoposeDataset')
class H5CachedEgoposeDataset(BaseDataset):
    """EgoPose Dataset with HDF5 caching for fast loading.

    This dataset pre-processes all JSON annotations into a single HDF5 cache file,
    dramatically reducing initialization time from minutes to seconds.

    Cache file structure (v1.0):
        - img_paths: (N,) string array of image paths
        - keypoints: (N, 1, 16, 2) float32 array of 2D keypoints
        - keypoint3d: (N, 1, 16, 3) float32 array of 3D keypoints
        - hmd_info: (N, 1, 9) float32 array of HMD preprocessed data
        - actions: (N,) string array of action labels

    Cache file structure (v2.0 with images):
        - All of the above, plus:
        - images: (N, 256, 256, 3) uint8 array of preprocessed RGB images
        - has_images: True (attribute)
        - img_size: 256 (attribute)

    Args:
        data_mode (str): Dataset mode. Default: 'topdown'.
        metainfo (dict, optional): Meta information for the dataset.
        data_root (str, optional): Root directory of the dataset.
        cache_file (str, optional): Path to H5 cache file. If None, will be
            auto-generated at data_root/annotations_cache.h5
        rebuild_cache (bool): Whether to rebuild cache even if it exists.
            Default: False.
        use_cached_images (bool): Whether to use cached images if available.
            Set to True to use pre-processed images from H5 cache for faster
            loading. Default: False.
        data_prefix (dict): Prefix for data path. Default: dict(img='').
        filter_cfg (dict, optional): Config for filtering data.
        indices (int or Sequence[int], optional): Indices to select.
        serialize_data (bool): Whether to serialize data. Default: True.
        pipeline (list): Processing pipeline. Default: [].
        test_mode (bool): Whether in test mode. Default: False.
        lazy_init (bool): Whether to use lazy init. Default: False.
        max_refetch (int): Max refetch times. Default: 1000.
        sample_interval (int): Sample interval. Default: 1.
    """

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100
    METAINFO: dict = dict(from_file=os.path.join(_CURRENT_DIR, 'egopose_info.py'))

    def __init__(self,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 cache_file: Optional[str] = None,
                 rebuild_cache: bool = False,
                 use_cached_images: bool = False,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 sample_interval: int = 1):
        """Initialize H5CachedEgoposeDataset."""

        self.data_root = data_root
        self.data_mode = data_mode
        self.sample_interval = sample_interval
        self.use_cached_images = use_cached_images

        # Determine cache file path
        if cache_file is None:
            self.cache_file = os.path.join(data_root, 'annotations_cache.h5')
        else:
            self.cache_file = cache_file

        self.rebuild_cache = rebuild_cache

        # Check if cache has images (will be set during cache loading)
        self._has_cached_images = False
        self._cached_img_size = 256

        # Load or build cache
        self._ensure_cache_exists()

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

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Adds metainfo fields (flip_indices, etc.) to data_info for transforms.

        Args:
            idx: Index of data sample.

        Returns:
            dict: Data info dict with metainfo fields added.
        """
        data_info = super().get_data_info(idx)

        # Add metainfo items required by transforms (e.g., RandomFlip)
        metainfo_keys = [
            'flip_indices', 'skeleton_links', 'upper_body_ids', 'lower_body_ids'
        ]

        for key in metainfo_keys:
            if key in self._metainfo and key not in data_info:
                data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def _ensure_cache_exists(self):
        """Ensure cache file exists, build if necessary."""
        if os.path.exists(self.cache_file) and not self.rebuild_cache:
            print_log(f'Loading cached annotations from {self.cache_file}',
                     logger='current', level=logging.INFO)
            return

        print_log(f'Building annotation cache at {self.cache_file}...',
                 logger='current', level=logging.INFO)
        self._build_cache()

    def _build_cache(self):
        """Build HDF5 cache from raw JSON annotations.

        This method:
        1. Recursively finds all JSON/image pairs in data_root
        2. Parses each JSON file and extracts annotations
        3. Computes HMD preprocessing data
        4. Saves everything to a single H5 file
        """
        # First, index all files
        print_log('Indexing dataset files...', logger='current', level=logging.INFO)
        index = self._index_dir(self.data_root)

        n_samples = len(index['json'])
        print_log(f'Found {n_samples} samples, parsing annotations...',
                 logger='current', level=logging.INFO)

        # Pre-allocate arrays
        img_paths = []
        keypoints = np.empty((n_samples, 1, 16, 2), dtype=np.float32)
        keypoint3d = np.empty((n_samples, 1, 16, 3), dtype=np.float32)
        hmd_info = np.empty((n_samples, 1, 9), dtype=np.float32)
        actions = []

        # Parse all annotations with progress bar
        for idx, (img_path, json_path) in enumerate(tqdm(
                zip(index['rgba'], index['json']),
                total=n_samples,
                desc='Building cache')):

            # Decode if bytes
            if isinstance(img_path, bytes):
                img_path = img_path.decode('utf8')
            if isinstance(json_path, bytes):
                json_path = json_path.decode('utf8')

            # Parse JSON
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract joint data
            joint_names = {j['name'].replace('mixamorig:', ''): jid
                          for jid, j in enumerate(data['joints'])}
            p2d_orig = np.array(data['pts2d_fisheye']).T
            p3d_orig = np.array(data['pts3d_fisheye']).T
            action = data['action']

            # Filter to skeleton joints
            p2d = np.empty([len(config.skel), 2], dtype=np.float32)
            p3d = np.empty([len(config.skel), 3], dtype=np.float32)

            for jid, j in enumerate(config.skel.keys()):
                p2d[jid] = p2d_orig[joint_names[j]]
                p3d[jid] = p3d_orig[joint_names[j]]

            # Convert to meters
            p3d /= self.CM_TO_M

            # Compute HMD info
            hmd = self._preprocess_hmd_data(p3d)

            # Store data
            img_paths.append(img_path)
            keypoints[idx, 0] = p2d
            keypoint3d[idx, 0] = p3d
            hmd_info[idx, 0] = hmd
            actions.append(action)

        # Save to HDF5
        print_log(f'Saving cache to {self.cache_file}...',
                 logger='current', level=logging.INFO)

        with h5py.File(self.cache_file, 'w') as hf:
            # String arrays need special handling
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset('img_paths', data=img_paths, dtype=dt)
            hf.create_dataset('actions', data=actions, dtype=dt)

            # Numeric arrays
            hf.create_dataset('keypoints', data=keypoints, dtype=np.float32)
            hf.create_dataset('keypoint3d', data=keypoint3d, dtype=np.float32)
            hf.create_dataset('hmd_info', data=hmd_info, dtype=np.float32)

            # Store metadata
            hf.attrs['n_samples'] = n_samples
            hf.attrs['version'] = '1.0'

        print_log(f'Cache built successfully! {n_samples} samples saved.',
                 logger='current', level=logging.INFO)

    def _preprocess_hmd_data(self, p3d: np.ndarray) -> np.ndarray:
        """Preprocess HMD data from 3D keypoints.

        Creates a local coordinate system from head and hand positions,
        then computes relative hand positions and distances.

        Args:
            p3d: (16, 3) array of 3D keypoints

        Returns:
            (9,) array of HMD preprocessed features:
                [right_local(3), left_local(3), hand_dist, right_dist, left_dist]
        """
        p3d = np.array(p3d)

        # Extract head and hand positions (indices from config.skel)
        head = p3d[0]       # Head
        right_hand = p3d[7]  # RightHand
        left_hand = p3d[4]   # LeftHand

        # Create local coordinate system
        midpoint = (right_hand + left_hand) / 2
        z_axis = midpoint - head
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0, 0, 1])

        hand_vector = right_hand - left_hand
        x_axis = np.cross(z_axis, hand_vector)
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            x_axis = np.array([1, 0, 0])

        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Transform to local coordinates
        right_local = np.dot(rotation_matrix.T, (right_hand - head))
        left_local = np.dot(rotation_matrix.T, (left_hand - head))

        # Compute distances
        hand_distance = np.linalg.norm(right_local - left_local)
        right_distance = np.linalg.norm(right_local)
        left_distance = np.linalg.norm(left_local)

        # Concatenate features
        preprocessed_hmd = np.concatenate([
            right_local, left_local,
            [hand_distance, right_distance, left_distance]
        ])

        return preprocessed_hmd.astype(np.float32)

    def _index_dir(self, path: str) -> Dict[str, List[str]]:
        """Recursively index all files in directory.

        Args:
            path: Root directory path

        Returns:
            Dict with 'rgba' and 'json' keys containing file path lists
        """
        indexed_paths = {k: [] for k in self.ROOT_DIRS}

        # Get subdirectories
        try:
            sub_dirs = next(os.walk(path))[1]
            sub_dirs.sort()
        except StopIteration:
            return indexed_paths

        # Check if this is a leaf directory with ROOT_DIRS
        if set(self.ROOT_DIRS) <= set(sub_dirs):
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                files = sorted([
                    os.path.join(d_path, f)
                    for f in os.listdir(d_path)
                    if os.path.isfile(os.path.join(d_path, f))
                ])
                indexed_paths[sub_dir] = files
            return indexed_paths

        # Recursively search subdirectories
        for sub_dir in sub_dirs:
            sub_indexed = self._index_dir(os.path.join(path, sub_dir))
            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(sub_indexed[r_dir])

        return indexed_paths

    def _transform_keypoints_for_cached_image(self, keypoints: np.ndarray,
                                               orig_w: int = 1280,
                                               orig_h: int = 800,
                                               img_size: int = 256) -> np.ndarray:
        """Transform 2D keypoints from original image coords to cached image coords.

        The cached images are created by:
        1. Center crop to square (min(h,w) x min(h,w))
        2. Resize to img_size x img_size

        For 1280x800 original -> 800x800 crop -> 256x256 resize:
        - crop_left = (1280 - 800) / 2 = 240
        - crop_top = 0
        - scale = 256 / 800 = 0.32

        Args:
            keypoints: (1, 16, 2) keypoints in original image coordinates
            orig_w: Original image width (default: 1280)
            orig_h: Original image height (default: 800)
            img_size: Cached image size (default: 256)

        Returns:
            Transformed keypoints in cached image coordinates
        """
        keypoints = keypoints.copy()
        min_dim = min(orig_h, orig_w)
        crop_left = (orig_w - min_dim) / 2
        crop_top = (orig_h - min_dim) / 2
        scale = img_size / min_dim

        # Transform: first subtract crop offset, then scale
        keypoints[..., 0] = (keypoints[..., 0] - crop_left) * scale
        keypoints[..., 1] = (keypoints[..., 1] - crop_top) * scale

        return keypoints

    def load_data_list(self) -> List[dict]:
        """Load annotations from HDF5 cache file.

        This is the key performance improvement - instead of parsing
        65k+ JSON files, we load pre-computed arrays from a single H5 file.

        When use_cached_images=True and cache has images:
        - Stores H5 cache path and image index for lazy loading via transform
        - Transforms 2D keypoints to match cached image coordinates
        - Uses bbox covering the full cached image
        - Images are loaded on-demand by LoadImageFromH5Cache transform

        Returns:
            List of annotation dicts
        """
        data_list = []

        with h5py.File(self.cache_file, 'r') as hf:
            n_samples = hf.attrs['n_samples']

            # Check if cache has images
            has_images = hf.attrs.get('has_images', False)
            img_size = hf.attrs.get('img_size', 256)

            # Load all annotation data at once (much faster than per-sample I/O)
            img_paths = hf['img_paths'][:]
            keypoints = hf['keypoints'][:]
            keypoint3d = hf['keypoint3d'][:]
            hmd_info = hf['hmd_info'][:]
            actions = hf['actions'][:]

            # Check if we can use cached images (lazy loading)
            if self.use_cached_images and has_images and 'images' in hf:
                self._has_cached_images = True
                self._cached_img_size = img_size
                print_log(f'H5 cache has {n_samples} images ({img_size}x{img_size}), '
                         f'will load lazily via LoadImageFromH5Cache transform',
                         logger='current', level=logging.INFO)

            # Check if cache is preprocessed (keypoints already transformed)
            # V2+ caches (version 2.x) have preprocessed keypoints even without explicit flag
            is_preprocessed = hf.attrs.get('preprocessed', False)
            cache_version = str(hf.attrs.get('version', '1.0'))
            if is_preprocessed or cache_version.startswith('2'):
                self._is_preprocessed = True
                self._cached_img_size = hf.attrs.get('img_size', 256)
                if cache_version.startswith('2') and not is_preprocessed:
                    print_log(f'V2 cache detected (version {cache_version}), '
                             f'keypoints already in {self._cached_img_size}x{self._cached_img_size} space',
                             logger='current', level=logging.INFO)

        # Apply sample interval
        indices = list(range(0, n_samples, self.sample_interval))

        # Determine bbox based on cache type
        if self._has_cached_images or getattr(self, '_is_preprocessed', False):
            # For preprocessed/cached images: bbox covers full image
            img_size = self._cached_img_size
            bbox = np.array([[0, 0, img_size, img_size]], dtype=np.float32)
        else:
            # For original images: use center crop (1000x800) from 1280x800
            bbox = np.array([[140, 0, 1140, 800]], dtype=np.float32)

        for i, idx in enumerate(indices):
            img_path = img_paths[idx] if isinstance(img_paths[idx], str) \
                       else img_paths[idx].decode('utf8')
            action = actions[idx] if isinstance(actions[idx], str) \
                     else actions[idx].decode('utf8')

            data_info = {
                'img_path': img_path,
                'keypoints': keypoints[idx].copy(),
                'keypoint3d': keypoint3d[idx].copy(),
                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'hmd_info': hmd_info[idx].copy(),
                'keypoints_visible': np.ones((1, 16), dtype=np.float32),
                'action': np.array([action])
            }

            # Add lazy loading info for cached images (H5 with embedded images)
            if self._has_cached_images:
                # Store H5 cache path and index for LoadImageFromH5Cache transform
                data_info['h5_cache_path'] = self.cache_file
                data_info['h5_img_idx'] = idx

                # Transform keypoints only for V1 caches (V2+ have preprocessed keypoints)
                if not getattr(self, '_is_preprocessed', False):
                    data_info['keypoints'] = self._transform_keypoints_for_cached_image(
                        keypoints[idx])
                # For V2+ preprocessed cache: keypoints are already in correct space

            data_list.append(data_info)

        mode_str = 'with lazy image loading from cache' if self._has_cached_images else 'from cache'
        print_log(f'Loaded {len(data_list)} samples {mode_str}',
                 logger='current', level=logging.INFO)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations. Default returns all data."""
        return self.data_list


@DATASETS.register_module(name='H5CachedEgoposeDataset_SegDepth')
class H5CachedEgoposeDataset_SegDepth(H5CachedEgoposeDataset):
    """H5 Cached EgoPose Dataset with Segmentation and Depth support.

    Extends base class to also cache depth and segmentation paths.
    """

    ROOT_DIRS = ['rgba', 'json', 'depth', 'objectId']

    def _build_cache(self):
        """Build cache including depth and segmentation paths."""
        print_log('Indexing dataset files...', logger='current', level=logging.INFO)
        index = self._index_dir(self.data_root)

        n_samples = len(index['json'])
        print_log(f'Found {n_samples} samples, parsing annotations...',
                 logger='current', level=logging.INFO)

        # Pre-allocate arrays
        img_paths = []
        depth_paths = []
        seg_paths = []
        keypoints = np.empty((n_samples, 1, 16, 2), dtype=np.float32)
        keypoint3d = np.empty((n_samples, 1, 16, 3), dtype=np.float32)
        hmd_info = np.empty((n_samples, 1, 9), dtype=np.float32)
        actions = []

        # Parse all annotations
        for idx, (img_path, json_path, depth_path, seg_path) in enumerate(tqdm(
                zip(index['rgba'], index['json'], index['depth'], index['objectId']),
                total=n_samples,
                desc='Building cache')):

            # Decode if bytes
            if isinstance(img_path, bytes):
                img_path = img_path.decode('utf8')
            if isinstance(json_path, bytes):
                json_path = json_path.decode('utf8')
            if isinstance(depth_path, bytes):
                depth_path = depth_path.decode('utf8')
            if isinstance(seg_path, bytes):
                seg_path = seg_path.decode('utf8')

            # Parse JSON
            with open(json_path, 'r') as f:
                data = json.load(f)

            joint_names = {j['name'].replace('mixamorig:', ''): jid
                          for jid, j in enumerate(data['joints'])}
            p2d_orig = np.array(data['pts2d_fisheye']).T
            p3d_orig = np.array(data['pts3d_fisheye']).T
            action = data['action']

            p2d = np.empty([len(config.skel), 2], dtype=np.float32)
            p3d = np.empty([len(config.skel), 3], dtype=np.float32)

            for jid, j in enumerate(config.skel.keys()):
                p2d[jid] = p2d_orig[joint_names[j]]
                p3d[jid] = p3d_orig[joint_names[j]]

            p3d /= self.CM_TO_M
            hmd = self._preprocess_hmd_data(p3d)

            img_paths.append(img_path)
            depth_paths.append(depth_path)
            seg_paths.append(seg_path)
            keypoints[idx, 0] = p2d
            keypoint3d[idx, 0] = p3d
            hmd_info[idx, 0] = hmd
            actions.append(action)

        # Save to HDF5
        print_log(f'Saving cache to {self.cache_file}...',
                 logger='current', level=logging.INFO)

        with h5py.File(self.cache_file, 'w') as hf:
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset('img_paths', data=img_paths, dtype=dt)
            hf.create_dataset('depth_paths', data=depth_paths, dtype=dt)
            hf.create_dataset('seg_paths', data=seg_paths, dtype=dt)
            hf.create_dataset('actions', data=actions, dtype=dt)

            hf.create_dataset('keypoints', data=keypoints, dtype=np.float32)
            hf.create_dataset('keypoint3d', data=keypoint3d, dtype=np.float32)
            hf.create_dataset('hmd_info', data=hmd_info, dtype=np.float32)

            hf.attrs['n_samples'] = n_samples
            hf.attrs['version'] = '1.0'
            hf.attrs['has_depth'] = True
            hf.attrs['has_seg'] = True

        print_log(f'Cache built successfully! {n_samples} samples saved.',
                 logger='current', level=logging.INFO)

    def load_data_list(self) -> List[dict]:
        """Load annotations including depth and segmentation paths."""
        data_list = []

        with h5py.File(self.cache_file, 'r') as hf:
            n_samples = hf.attrs['n_samples']

            img_paths = hf['img_paths'][:]
            depth_paths = hf['depth_paths'][:]
            seg_paths = hf['seg_paths'][:]
            keypoints = hf['keypoints'][:]
            keypoint3d = hf['keypoint3d'][:]
            hmd_info = hf['hmd_info'][:]
            actions = hf['actions'][:]

        indices = range(0, n_samples, self.sample_interval)
        # Use center crop (1000x800) from original 1280x800 image
        bbox = np.array([[140, 0, 1140, 800]], dtype=np.float32)

        for idx in indices:
            data_info = {
                'img_path': img_paths[idx] if isinstance(img_paths[idx], str)
                           else img_paths[idx].decode('utf8'),
                'depth_path': depth_paths[idx] if isinstance(depth_paths[idx], str)
                             else depth_paths[idx].decode('utf8'),
                'seg_path': seg_paths[idx] if isinstance(seg_paths[idx], str)
                           else seg_paths[idx].decode('utf8'),
                'keypoints': keypoints[idx],
                'keypoint3d': keypoint3d[idx],
                'bbox': bbox.copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'hmd_info': hmd_info[idx],
                'keypoints_visible': np.ones((1, 16), dtype=np.float32),
                'action': np.array([actions[idx] if isinstance(actions[idx], str)
                                   else actions[idx].decode('utf8')])
            }
            data_list.append(data_info)

        print_log(f'Loaded {len(data_list)} samples from cache (with depth & seg)',
                 logger='current', level=logging.INFO)

        return data_list
