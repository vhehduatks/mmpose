# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import h5py
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromH5(object):
    """Load an image from HDF5 file (for mo2cap2 dataset).

    This transform loads images directly from H5 chunk files,
    which is much faster than loading individual image files.

    Required Keys:
        - h5_chunk_path: Path to H5 chunk file
        - h5_local_idx: Index within the chunk
        - use_zoom (optional): Whether to use ZoomImages

    Modified Keys:
        - img: (H, W, 3) uint8 array in HWC BGR format
        - img_shape
        - ori_shape

    Args:
        to_float32 (bool): Whether to convert to float32. Default: False.
        lazy_load_keypoints (bool): Whether to also load keypoints from H5.
            Use with H5Mo2Cap2Dataset_Lazy. Default: False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 lazy_load_keypoints: bool = False):
        self.to_float32 = to_float32
        self.lazy_load_keypoints = lazy_load_keypoints
        self._h5_cache = {}  # Cache open file handles

    def _get_h5_handle(self, h5_path: str) -> h5py.File:
        """Get cached H5 file handle."""
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path]

    def transform(self, results: dict) -> Optional[dict]:
        """Load image from H5 file.

        Args:
            results (dict): Result dict from dataset

        Returns:
            dict: Updated result dict with loaded image
        """
        h5_path = results.get('h5_chunk_path')
        local_idx = results.get('h5_local_idx')
        use_zoom = results.get('use_zoom', False)

        if h5_path is None or local_idx is None:
            raise KeyError('h5_chunk_path and h5_local_idx are required '
                          'for LoadImageFromH5')

        # Open H5 file (cached)
        hf = self._get_h5_handle(h5_path)

        # Select image dataset
        img_key = 'ZoomImages' if use_zoom else 'Images'

        # Load image: (3, 256, 256) -> (256, 256, 3)
        img = hf[img_key][local_idx]  # (C, H, W) uint8
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # Convert RGB to BGR (OpenCV format)
        img = img[:, :, ::-1].copy()

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        # Lazy load keypoints if needed
        if self.lazy_load_keypoints and results.get('lazy_load', False):
            mm_to_m = 1000

            annot2d = hf['Annot2D'][local_idx:local_idx+1]  # (1, 15, 2)
            annot3d = hf['Annot3D'][local_idx:local_idx+1]  # (1, 15, 3)

            # Convert to meters and make root-relative
            annot3d = annot3d / mm_to_m
            annot3d = annot3d - annot3d[:, 0:1, :]

            results['keypoints'] = annot2d.astype(np.float32)
            results['keypoint3d'] = annot3d.astype(np.float32)

            # Compute HMD info
            p3d = annot3d[0]
            results['hmd_info'] = self._preprocess_hmd_data(p3d)[np.newaxis, :]

        return results

    def _preprocess_hmd_data(self, p3d: np.ndarray) -> np.ndarray:
        """Preprocess HMD data from 3D keypoints."""
        head = p3d[0]
        right_hand = p3d[3]
        left_hand = p3d[6]

        midpoint = (right_hand + left_hand) / 2
        z_axis = midpoint - head
        z_norm = np.linalg.norm(z_axis)
        z_axis = z_axis / z_norm if z_norm > 1e-6 else np.array([0, 0, 1])

        hand_vector = right_hand - left_hand
        x_axis = np.cross(z_axis, hand_vector)
        x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm if x_norm > 1e-6 else np.array([1, 0, 0])

        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        right_local = np.dot(rotation_matrix.T, (right_hand - head))
        left_local = np.dot(rotation_matrix.T, (left_hand - head))

        hand_distance = np.linalg.norm(right_local - left_local)
        right_distance = np.linalg.norm(right_local)
        left_distance = np.linalg.norm(left_local)

        return np.concatenate([
            right_local, left_local,
            [hand_distance, right_distance, left_distance]
        ]).astype(np.float32)

    def __call__(self, results: dict) -> Optional[dict]:
        """Call transform method."""
        return self.transform(results)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'lazy_load_keypoints={self.lazy_load_keypoints})')


@TRANSFORMS.register_module()
class LoadImageFromH5Cache(object):
    """Load an image from H5 cache file (for EgoPose dataset with cached images).

    This transform loads preprocessed images directly from H5 cache files,
    which is faster than loading individual image files from disk.

    Required Keys:
        - h5_cache_path: Path to H5 cache file
        - h5_img_idx: Index of the image within the cache

    Modified Keys:
        - img: (H, W, 3) uint8 array in HWC BGR format
        - img_shape
        - ori_shape

    Args:
        to_float32 (bool): Whether to convert to float32. Default: False.
    """

    # Class-level cache for preloaded images (populated by dataset)
    _preloaded_images = {}

    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32
        self._h5_cache = {}  # Cache open file handles

    def _get_h5_handle(self, h5_path: str) -> h5py.File:
        """Get cached H5 file handle."""
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path]

    def transform(self, results: dict) -> Optional[dict]:
        """Load image from H5 cache file.

        Args:
            results (dict): Result dict from dataset

        Returns:
            dict: Updated result dict with loaded image
        """
        h5_path = results.get('h5_cache_path')
        img_idx = results.get('h5_img_idx')

        if h5_path is None or img_idx is None:
            raise KeyError('h5_cache_path and h5_img_idx are required '
                          'for LoadImageFromH5Cache')

        # Load image: (H, W, 3) uint8 RGB
        if h5_path in self._preloaded_images:
            # Fast path: read from preloaded in-memory array
            img = self._preloaded_images[h5_path][img_idx]
        else:
            # Fallback: per-sample H5 random access
            hf = self._get_h5_handle(h5_path)
            img = hf['images'][img_idx]

        # Convert RGB to BGR (OpenCV format)
        img = img[:, :, ::-1].copy()

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

    def __call__(self, results: dict) -> Optional[dict]:
        """Make the transform callable."""
        return self.transform(results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'


@TRANSFORMS.register_module()
class LoadDepthFromH5Cache(object):
    """Load a depth map from H5 cache file (for EgoPose dataset with cached depth).

    Loads single-channel depth stored as uint8 (256x256) from the 'depths'
    dataset in the H5 cache. Normalizes to float32 [0, 1].

    Required Keys:
        - h5_cache_path: Path to H5 cache file
        - h5_img_idx: Index of the sample within the cache

    Modified Keys:
        - depth_map: (1, H, W) float32 tensor-ready array

    Args:
        normalize (bool): Whether to normalize uint8 to [0, 1]. Default: True.
    """

    _preloaded_depths = {}

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._h5_cache = {}

    def _get_h5_handle(self, h5_path: str):
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path]

    def transform(self, results: dict) -> Optional[dict]:
        h5_path = results.get('h5_cache_path')
        img_idx = results.get('h5_img_idx')

        if h5_path is None or img_idx is None:
            raise KeyError('h5_cache_path and h5_img_idx are required')

        if h5_path in self._preloaded_depths:
            depth = self._preloaded_depths[h5_path][img_idx]
        else:
            hf = self._get_h5_handle(h5_path)
            if 'depths' not in hf:
                # No depth available — provide zeros
                h, w = results.get('img_shape', (256, 256))
                results['depth_map'] = np.zeros((1, h, w), dtype=np.float32)
                return results
            depth = hf['depths'][img_idx]

        # depth: (H, W) uint8
        depth = depth.astype(np.float32)
        if self.normalize:
            depth = depth / 255.0

        # Add channel dim: (1, H, W)
        results['depth_map'] = depth[np.newaxis, :, :]
        return results

    def __call__(self, results: dict) -> Optional[dict]:
        return self.transform(results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(normalize={self.normalize})'


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results
