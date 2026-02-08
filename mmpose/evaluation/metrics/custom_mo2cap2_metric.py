# Copyright (c) OpenMMLab. All rights reserved.
"""
Mo2Cap2 3D Pose Evaluation Metric

Matches official MATLAB evaluation protocol (mo2cap2_eval.m):
- Skeleton rescaling to reference bone lengths
- Procrustes alignment WITHOUT scaling
- Per-action breakdown using official frame ranges

Reports: Full Body, Upper Body, Lower Body MPJPE (and Per-Joint)
"""
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from . import mo2cap2_evaluate


@METRICS.register_module()
class CustomMo2Cap2Metric(BaseMetric):
    """Mo2Cap2 3D Pose Evaluation Metric (Official Protocol).

    Evaluates 3D pose predictions using MPJPE following the official
    Mo2Cap2 evaluation protocol:
    - Skeleton rescaling to reference bone lengths
    - Procrustes alignment WITHOUT scaling (matches MATLAB)
    - Per-action breakdown using official frame ranges

    Reports:
    - Full Body MPJPE (all 15 joints)
    - Upper Body MPJPE (joints 0-6: Neck + Arms)
    - Lower Body MPJPE (joints 7-14: Legs)
    - Per-Joint MPJPE

    Args:
        use_action: Whether to compute per-action metrics. Defaults to True.
        collect_device: Device for collecting results. Defaults to 'cpu'.
        prefix: Metric prefix. Defaults to 'mo2cap2'.
    """

    default_prefix: Optional[str] = 'mo2cap2'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_action: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.use_action = use_action
        self._dataset_meta = None

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process predictions and ground truth for each batch.

        Extracts 3D keypoints, frame indices, and sequence names
        for proper per-action evaluation.

        Args:
            data_batch: Batch of input data (unused).
            data_samples: List of data samples containing predictions and GT.
        """
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    f'`pred_instances` required in {self.__class__.__name__}')

            # Extract prediction
            pred = {
                'img_id': data_sample['img_id'],
                'keypoint3d': data_sample['pred_instances']['keypoint_3d'],
            }

            # Extract ground truth
            gt = {
                'img_id': data_sample['img_id'],
                'keypoint3d': data_sample['gt_instance_labels']['keypoint3d'],
            }

            # For official per-action evaluation
            if self.use_action:
                # Frame index within the sequence (0-indexed)
                gt['frame_idx'] = data_sample.get('frame_idx', -1)
                # Sequence name ('olek_outdoor' or 'weipeng_studio')
                gt['sequence_name'] = data_sample.get('sequence_name',
                                                       data_sample.get('action', 'unknown'))

            self.results.append((pred, gt))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute Mo2Cap2 3D pose evaluation metrics.

        Uses official MATLAB protocol:
        - Skeleton rescaling + Procrustes (no scaling)
        - Per-action breakdown using frame ranges

        Args:
            results: List of (pred, gt) tuples collected from process().

        Returns:
            Dictionary of metric names and values (MPJPE in mm).
        """
        logger = MMLogger.get_current_instance()
        logger.info(f'Evaluating {self.__class__.__name__} (Official Protocol)...')

        # Initialize evaluators
        eval_body = mo2cap2_evaluate.EvalBody()
        eval_upper = mo2cap2_evaluate.EvalUpperBody()
        eval_lower = mo2cap2_evaluate.EvalLowerBody()
        eval_per_joint = mo2cap2_evaluate.EvalPerJoint()

        # Collect predictions and ground truth
        preds, gts = zip(*results)

        pred_keypoints = [p['keypoint3d'] for p in preds]
        gt_keypoints = [g['keypoint3d'] for g in gts]

        # For per-action evaluation
        frame_indices = None
        sequence_names = None
        if self.use_action:
            frame_indices = [g.get('frame_idx', -1) for g in gts]
            sequence_names = [g.get('sequence_name', 'unknown') for g in gts]

        # Stack tensors (squeeze removes batch dim if present)
        pred_batch = torch.stack(pred_keypoints).squeeze(dim=1)
        gt_batch = torch.stack(gt_keypoints).squeeze(dim=1)

        # Run evaluation (Full Body, Upper Body, Lower Body)
        eval_body.eval(pred_batch, gt_batch,
                      use_action_=self.use_action,
                      frame_indices=frame_indices,
                      sequence_names=sequence_names)
        eval_upper.eval(pred_batch, gt_batch,
                       use_action_=self.use_action,
                       frame_indices=frame_indices,
                       sequence_names=sequence_names)
        eval_lower.eval(pred_batch, gt_batch,
                       use_action_=self.use_action,
                       frame_indices=frame_indices,
                       sequence_names=sequence_names)
        eval_per_joint.eval(pred_batch, gt_batch)

        # Get results
        body_results = eval_body.get_results()
        upper_results = eval_upper.get_results()
        lower_results = eval_lower.get_results()
        per_joint_results = eval_per_joint.get_results()

        # Get environment results
        body_env_results = eval_body.get_environment_results()
        upper_env_results = eval_upper.get_environment_results()
        lower_env_results = eval_lower.get_environment_results()

        # Log detailed results
        logger.info(f'Full Body: {body_results}')
        logger.info(f'Upper Body: {upper_results}')
        logger.info(f'Lower Body: {lower_results}')
        logger.info(f'Per Joint MPJPE (mm): {per_joint_results}')

        # Log action breakdown if available
        if self.use_action:
            logger.info('\n' + mo2cap2_evaluate.get_action_breakdown_summary(body_results))
            # Log environment breakdown
            logger.info('\n' + mo2cap2_evaluate.get_environment_breakdown_summary(body_env_results))

        # Build output metrics dictionary
        eval_results = OrderedDict()

        # Add Full Body, Upper Body, Lower Body metrics (overall + per-action)
        for part_name, part_results in [
            ('Full Body', body_results),
            ('Upper Body', upper_results),
            ('Lower Body', lower_results),
        ]:
            for action_name, metrics in part_results.items():
                metric_key = f'{part_name}_{action_name}_mpjpe'
                eval_results[metric_key] = metrics['mpjpe']

        # Add environment-specific metrics
        for part_name, env_results in [
            ('Full Body', body_env_results),
            ('Upper Body', upper_env_results),
            ('Lower Body', lower_env_results),
        ]:
            for env_name, metrics in env_results.items():
                metric_key = f'{part_name}_{env_name}_mpjpe'
                eval_results[metric_key] = metrics['mpjpe']

        return eval_results
