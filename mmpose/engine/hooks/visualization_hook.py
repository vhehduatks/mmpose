# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class PoseVisualizationHook(Hook):
    """Pose Estimation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        enable (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        train_interval: int = 0,  # 0 means disabled, >0 enables training visualization
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
        # Normalization parameters for denormalization (defaults to ImageNet)
        img_mean: Optional[list] = None,
        img_std: Optional[list] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.train_interval = train_interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
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
        # Store normalization parameters (RGB order)
        # Defaults to ImageNet if not specified
        self.img_mean = img_mean if img_mean is not None else [123.675, 116.28, 103.53]
        self.img_std = img_std if img_std is not None else [58.395, 57.12, 57.375]
    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                         outputs: dict) -> None:
        """Run after every ``self.train_interval`` training iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict): Data from dataloader.
            outputs (dict): Training outputs (losses).
        """
        if not self.enable or self.train_interval <= 0:
            return

        if runner.iter % self.train_interval != 0:
            return

        # Set dataset meta
        if hasattr(runner, 'val_evaluator') and runner.val_evaluator is not None:
            self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # Get image from data_batch
        data_samples = data_batch.get('data_samples', [])
        if not data_samples:
            return

        # Get image - try img_path first, then try to get from inputs
        import torch
        import numpy as np
        img = None
        img_path = data_samples[0].get('img_path', '')

        # Check if img_path is a real file (not H5 virtual path)
        is_real_file = img_path and not img_path.startswith('h5://') and os.path.exists(img_path)

        if is_real_file:
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        elif img_path.startswith('h5://'):
            # Load raw image directly from H5 file (Mo2Cap2 training data)
            try:
                import h5py
                h5_chunk_path = data_samples[0].metainfo.get('h5_chunk_path')
                h5_local_idx = data_samples[0].metainfo.get('h5_local_idx')
                if h5_chunk_path and h5_local_idx is not None:
                    with h5py.File(h5_chunk_path, 'r') as hf:
                        # Images stored as (C, H, W) RGB in H5
                        img_data = hf['Images'][h5_local_idx]  # (3, 256, 256)
                        img = np.transpose(img_data, (1, 2, 0))  # (H, W, C)
                        # H5 stores RGB, convert to BGR for OpenCV compatibility
                        img = img[:, :, ::-1].copy()
            except Exception as e:
                img = None

        # Fallback: denormalize from preprocessed inputs
        if img is None and 'inputs' in data_batch:
            inputs = data_batch['inputs']
            # Handle both list and tensor inputs
            if isinstance(inputs, list) and len(inputs) > 0:
                img_tensor = inputs[0]
            elif isinstance(inputs, torch.Tensor) and len(inputs) > 0:
                img_tensor = inputs[0]
            else:
                img_tensor = None

            if img_tensor is not None:
                # Denormalize using configured normalization parameters (RGB order)
                device = img_tensor.device
                mean = torch.tensor(self.img_mean).reshape(3, 1, 1).to(device)
                std = torch.tensor(self.img_std).reshape(3, 1, 1).to(device)
                img_tensor = img_tensor.float()
                img_tensor = img_tensor * std + mean
                img_tensor = img_tensor.cpu()
                img = img_tensor.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV compatibility
                img = img[:, :, ::-1].copy()

        if img is None:
            return

        # Run inference to get predictions
        import torch
        with torch.no_grad():
            runner.model.eval()
            preds = runner.model.val_step(data_batch)
            runner.model.train()

        if not preds:
            return

        data_sample = merge_data_samples([preds[0]])

        # Transform pred keypoints from original image space to display space
        if (hasattr(data_sample, 'pred_instances')
                and 'keypoints' in data_sample.pred_instances
                and img is not None):
            import numpy as np
            kpts = data_sample.pred_instances.keypoints.copy()
            input_scale = data_sample.metainfo.get('input_scale', None)
            if input_scale is not None:
                ori_w, ori_h = float(input_scale[0]), float(input_scale[1])
                disp_h, disp_w = img.shape[:2]
                kpts[..., 0] *= disp_w / ori_w
                kpts[..., 1] *= disp_h / ori_h
            data_sample.pred_instances.transformed_keypoints = kpts

        # Use unique name with step for WandB tracking
        img_name = f'train_img_{runner.iter}'
        self._visualizer.add_datasample(
            img_name,
            img,
            data_sample=data_sample,
            draw_gt=True,
            draw_bbox=False,
            show=self.show,
            wait_time=self.wait_time,
            kpt_thr=self.kpt_thr,
            step=runner.iter)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Get the transformed image from inputs (256x256) to match predicted
        # keypoint coordinates. data_batch['inputs'] comes from the dataloader
        # BEFORE PoseDataPreprocessor runs, so it may be either:
        #   - uint8 (raw pixels, no normalization applied yet)
        #   - float32 (already normalized by the preprocessor)
        # We detect which case and handle accordingly.
        import torch
        import numpy as np
        img = None

        if 'inputs' in data_batch:
            inputs = data_batch['inputs']
            if isinstance(inputs, list) and len(inputs) > 0:
                img_tensor = inputs[0]
            elif isinstance(inputs, torch.Tensor) and len(inputs) > 0:
                img_tensor = inputs[0]
            else:
                img_tensor = None

            if img_tensor is not None:
                if img_tensor.dtype == torch.uint8:
                    # Raw uint8 pixels (RGB from PackPoseInputs) — no denorm needed
                    img = img_tensor.cpu().permute(1, 2, 0).numpy()
                    # Convert RGB to BGR for OpenCV compatibility
                    img = img[:, :, ::-1].copy()
                else:
                    # Float tensor — was normalized, denormalize back
                    device = img_tensor.device
                    mean = torch.tensor(self.img_mean).reshape(3, 1, 1).to(device)
                    std = torch.tensor(self.img_std).reshape(3, 1, 1).to(device)
                    img_tensor = img_tensor.float()
                    img_tensor = img_tensor * std + mean
                    img_tensor = img_tensor.cpu()
                    img = img_tensor.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV compatibility
                    img = img[:, :, ::-1].copy()

        # Fallback to loading from disk if inputs not available
        if img is None:
            img_path = data_batch['data_samples'][0].get('img_path')
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='bgr')

        data_sample = outputs[0]

        # revert the heatmap on the original image
        data_sample = merge_data_samples([data_sample])

        # Transform pred keypoints from original image space to display image
        # space. The model decodes heatmaps back to the original image coords
        # (e.g. 1920x1080) using input_center/input_scale, but we display the
        # preprocessed 256x256 tensor. The visualizer checks for
        # 'transformed_keypoints' first (see _draw_instances_kpts), so we
        # add scaled coordinates that match the display image size.
        if (hasattr(data_sample, 'pred_instances')
                and 'keypoints' in data_sample.pred_instances
                and img is not None):
            kpts = data_sample.pred_instances.keypoints.copy()
            # Get original image dimensions from metainfo
            input_scale = data_sample.metainfo.get('input_scale', None)
            if input_scale is not None:
                ori_w, ori_h = float(input_scale[0]), float(input_scale[1])
                disp_h, disp_w = img.shape[:2]
                kpts[..., 0] *= disp_w / ori_w
                kpts[..., 1] *= disp_h / ori_h
            data_sample.pred_instances.transformed_keypoints = kpts

        if total_curr_iter % self.interval == 0:
            # Use unique name with step for WandB tracking
            if self.show:
                img_name = os.path.basename(img_path)
            else:
                img_name = f'val_img_{total_curr_iter}'
            self._visualizer.add_datasample(
                img_name,
                img,
                data_sample=data_sample,
                draw_gt=True,
                draw_bbox=False,
                # draw_heatmap=False,
                show=self.show,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        # Get transformed images from inputs (256x256) to match predicted keypoint coordinates
        import torch
        import numpy as np
        batch_imgs = []

        if 'inputs' in data_batch:
            inputs = data_batch['inputs']
            for i in range(len(outputs)):
                if isinstance(inputs, list) and i < len(inputs):
                    img_tensor = inputs[i]
                elif isinstance(inputs, torch.Tensor) and i < len(inputs):
                    img_tensor = inputs[i]
                else:
                    img_tensor = None

                if img_tensor is not None:
                    if img_tensor.dtype == torch.uint8:
                        # Raw uint8 pixels — no denorm needed
                        img = img_tensor.cpu().permute(1, 2, 0).numpy()
                        img = img[:, :, ::-1].copy()
                    else:
                        # Float tensor — denormalize
                        device = img_tensor.device
                        mean = torch.tensor(self.img_mean).reshape(3, 1, 1).to(device)
                        std = torch.tensor(self.img_std).reshape(3, 1, 1).to(device)
                        img_tensor = img_tensor.float()
                        img_tensor = img_tensor * std + mean
                        img_tensor = img_tensor.cpu()
                        img = img_tensor.permute(1, 2, 0).numpy()
                        img = np.clip(img, 0, 255).astype(np.uint8)
                        img = img[:, :, ::-1].copy()
                    batch_imgs.append(img)
                else:
                    batch_imgs.append(None)

        for idx, data_sample in enumerate(outputs):
            self._test_index += 1

            img_path = data_sample.get('img_path')

            # Use transformed image if available, otherwise load from disk
            if idx < len(batch_imgs) and batch_imgs[idx] is not None:
                img = batch_imgs[idx]
            else:
                img_bytes = fileio.get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='bgr')

            data_sample = merge_data_samples([data_sample])

            # Transform pred keypoints from original image space to display space
            if (hasattr(data_sample, 'pred_instances')
                    and 'keypoints' in data_sample.pred_instances
                    and img is not None):
                kpts = data_sample.pred_instances.keypoints.copy()
                input_scale = data_sample.metainfo.get('input_scale', None)
                if input_scale is not None:
                    ori_w, ori_h = float(input_scale[0]), float(input_scale[1])
                    disp_h, disp_w = img.shape[:2]
                    kpts[..., 0] *= disp_w / ori_w
                    kpts[..., 1] *= disp_h / ori_h
                data_sample.pred_instances.transformed_keypoints = kpts

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit(
                    '.', 1)
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
                # draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index)
