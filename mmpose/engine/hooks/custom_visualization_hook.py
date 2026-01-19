# Copyright (c) OpenMMLab. All rights reserved.
"""Custom Visualization Hook that supports H5 cached images."""
import os
import warnings
from typing import Optional, Sequence

import h5py
import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class H5CacheVisualizationHook(Hook):
    """Pose Visualization Hook with H5 cache support.

    This hook can load images from:
    1. H5 cache file (if h5_cache_path and h5_img_idx are in metainfo)
    2. Original image path (fallback)

    Args:
        enable (bool): Whether to draw prediction results. Default: False.
        interval (int): Visualization interval. Default: 50.
        kpt_thr (float): Keypoint score threshold. Default: 0.3.
        show (bool): Whether to display the image. Default: False.
        wait_time (float): Display wait time in seconds. Default: 0.
        out_dir (str, optional): Output directory for test images.
        backend_args (dict, optional): Backend arguments for file I/O.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args
        self._h5_cache = {}  # Cache for H5 file handles

    def _get_h5_handle(self, h5_path: str) -> h5py.File:
        """Get cached H5 file handle."""
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path]

    def _load_image(self, data_sample: PoseDataSample) -> np.ndarray:
        """Load image from H5 cache or disk.

        Args:
            data_sample: Data sample containing metainfo

        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        metainfo = data_sample.metainfo

        # Try H5 cache first
        h5_cache_path = metainfo.get('h5_cache_path')
        h5_img_idx = metainfo.get('h5_img_idx')

        if h5_cache_path is not None and h5_img_idx is not None:
            try:
                hf = self._get_h5_handle(h5_cache_path)
                # Load image from H5 cache (already RGB)
                img = hf['images'][h5_img_idx]
                return np.array(img)
            except Exception as e:
                warnings.warn(f'Failed to load from H5 cache: {e}, '
                              'falling back to disk')

        # Fallback to disk
        img_path = metainfo.get('img_path')
        if img_path:
            try:
                img_bytes = fileio.get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                return img
            except Exception as e:
                warnings.warn(f'Failed to load image from {img_path}: {e}')

        # Return placeholder if all else fails
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after validation iterations."""
        if not self.enable:
            return

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        total_curr_iter = runner.iter + batch_idx

        if total_curr_iter % self.interval != 0:
            return

        # Get first sample
        data_sample = outputs[0]
        img = self._load_image(data_sample)

        # Get img_path for naming
        img_path = data_sample.metainfo.get('img_path', 'val_img')

        # Merge data samples for visualization
        data_sample = merge_data_samples([data_sample])

        self._visualizer.add_datasample(
            os.path.basename(img_path) if self.show else 'val_img',
            img,
            data_sample=data_sample,
            draw_gt=True,
            draw_bbox=False,
            show=self.show,
            wait_time=self.wait_time,
            kpt_thr=self.kpt_thr,
            step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after testing iterations."""
        if not self.enable:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img = self._load_image(data_sample)
            img_path = data_sample.metainfo.get('img_path', 'test_img.png')

            data_sample = merge_data_samples([data_sample])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit('.', 1)
                index = len([
                    fname for fname in os.listdir(self.out_dir)
                    if fname.startswith(out_file_name)
                ])
                out_file = f'{out_file_name}_{index}.{postfix}'
                out_file = os.path.join(self.out_dir, out_file)

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=True,
                draw_bbox=False,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index)

    def __del__(self):
        """Close H5 file handles."""
        for hf in self._h5_cache.values():
            try:
                hf.close()
            except Exception:
                pass
