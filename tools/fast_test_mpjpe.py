#!/usr/bin/env python
"""
Fast MPJPE Test Script

Evaluates Full Body, Upper Body, Lower Body MPJPE without mmengine Runner overhead.
Much faster than tools/test.py for quick evaluation.

Usage:
    python tools/fast_test_mpjpe.py <config> <checkpoint> [--batch-size 128]
"""

import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import DefaultScope


def compute_mpjpe(pred, gt, joint_sel=None):
    """Compute MPJPE in mm.

    Args:
        pred: (N, 16, 3) predicted 3D keypoints
        gt: (N, 16, 3) ground truth 3D keypoints
        joint_sel: list of joint indices to evaluate (None = all)

    Returns:
        mean MPJPE in mm
    """
    if joint_sel is not None:
        pred = pred[:, joint_sel, :]
        gt = gt[:, joint_sel, :]

    # MPJPE = mean of L2 distance per joint, converted to mm
    error = np.sqrt(np.sum((pred - gt) ** 2, axis=2))  # (N, num_joints)
    mpjpe = np.mean(error) * 1000  # meters to mm
    return mpjpe


def main():
    parser = argparse.ArgumentParser(description='Fast MPJPE Test')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', default='cuda:0', help='Device')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Initialize mmpose scope
    with DefaultScope.overwrite_default_scope('mmpose'):
        import mmpose.datasets
        import mmpose.codecs
        import mmpose.models
        from mmpose.registry import DATASETS, MODELS
        from mmpose.datasets.datasets.body3d.custom_egopose_dataset_h5cache import H5CachedEgoposeDataset

        print(f'Loading dataset from cache...')
        start = time.time()
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        print(f'Dataset loaded: {len(dataset)} samples in {time.time() - start:.2f}s')

        # Build model
        print(f'Loading model from {args.checkpoint}...')
        model = MODELS.build(cfg.model)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

        model = model.to(args.device)
        model.eval()
        print(f'Model loaded')

        # Collect predictions and ground truth
        all_pred_3d = []
        all_gt_3d = []
        all_actions = []

        # Create simple dataloader
        from torch.utils.data import DataLoader
        from mmengine.dataset import pseudo_collate

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Single process is fastest for this dataset
            collate_fn=pseudo_collate,
            shuffle=False
        )

        print(f'\nRunning inference...')
        start = time.time()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc='Testing')):
                # batch_data is a dict: {'inputs': [tensor, ...], 'data_samples': [PoseDataSample, ...]}
                inputs_list = batch_data['inputs']
                data_samples_list = batch_data['data_samples']

                # Collect ground truth
                gt_3d_batch = []
                actions_batch = []
                for data_sample in data_samples_list:
                    gt_3d = data_sample.gt_instance_labels.keypoint3d
                    gt_3d_batch.append(gt_3d)

                    if hasattr(data_sample.gt_instances, 'action'):
                        actions_batch.append(data_sample.gt_instances.action[0])

                # Stack inputs
                inputs = torch.stack(inputs_list).to(args.device)

                # Forward pass
                outputs = model.test_step({'inputs': inputs, 'data_samples': data_samples_list})

                # Collect predictions
                for output in outputs:
                    pred_3d = output.pred_instances.keypoint_3d.cpu().numpy()
                    all_pred_3d.append(pred_3d.squeeze())

                # Collect ground truth
                for gt in gt_3d_batch:
                    if isinstance(gt, torch.Tensor):
                        gt = gt.cpu().numpy()
                    all_gt_3d.append(gt.squeeze())

                all_actions.extend(actions_batch)

        elapsed = time.time() - start
        print(f'\nInference completed in {elapsed:.2f}s ({len(dataset)/elapsed:.1f} samples/sec)')

        # Stack all predictions and ground truth
        all_pred_3d = np.array(all_pred_3d)  # (N, 16, 3)
        all_gt_3d = np.array(all_gt_3d)      # (N, 16, 3)

        print(f'\nPredictions shape: {all_pred_3d.shape}')
        print(f'Ground truth shape: {all_gt_3d.shape}')

        # Define joint indices for body parts (baseline mode)
        upper_body_joints = [0, 1, 2, 3, 4, 5, 6, 7]  # Head, Neck, Arms
        lower_body_joints = [8, 9, 10, 11, 12, 13, 14, 15]  # Legs

        # Compute MPJPE
        full_body_mpjpe = compute_mpjpe(all_pred_3d, all_gt_3d)
        upper_body_mpjpe = compute_mpjpe(all_pred_3d, all_gt_3d, upper_body_joints)
        lower_body_mpjpe = compute_mpjpe(all_pred_3d, all_gt_3d, lower_body_joints)

        # Per-joint MPJPE
        per_joint_error = np.sqrt(np.sum((all_pred_3d - all_gt_3d) ** 2, axis=2))
        per_joint_mpjpe = np.mean(per_joint_error, axis=0) * 1000

        # Print results
        print('\n' + '='*60)
        print('MPJPE Results (mm)')
        print('='*60)
        print(f'Full Body:  {full_body_mpjpe:.2f} mm')
        print(f'Upper Body: {upper_body_mpjpe:.2f} mm')
        print(f'Lower Body: {lower_body_mpjpe:.2f} mm')
        print('='*60)

        # Print per-joint results
        joint_names = [
            'Head', 'Neck', 'LeftArm', 'LeftForeArm',
            'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
            'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
        ]

        print('\nPer-Joint MPJPE (mm):')
        for i, (name, mpjpe) in enumerate(zip(joint_names, per_joint_mpjpe)):
            print(f'  {name:15s}: {mpjpe:.2f}')

        # Per-action evaluation if actions available
        if all_actions:
            print('\n' + '='*60)
            print('Per-Action Results')
            print('='*60)

            unique_actions = list(set(all_actions))
            unique_actions.sort()

            for action in unique_actions:
                indices = [i for i, a in enumerate(all_actions) if a == action]
                action_pred = all_pred_3d[indices]
                action_gt = all_gt_3d[indices]
                action_mpjpe = compute_mpjpe(action_pred, action_gt)
                print(f'{action:30s}: {action_mpjpe:.2f} mm ({len(indices)} samples)')

        return {
            'full_body': full_body_mpjpe,
            'upper_body': upper_body_mpjpe,
            'lower_body': lower_body_mpjpe,
            'per_joint': per_joint_mpjpe
        }


if __name__ == '__main__':
    main()
