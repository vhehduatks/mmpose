#!/usr/bin/env python3
"""
Visualize Mo2Cap2 V2 Cache Dataset

Train set: 2D photos, 2D joints, 3D joints (from H5 chunks)
Test set: 2D photos, 3D joints only (from JPG + MAT files)

Usage:
    python my_code/visualization/visualize_mo2cap2_v2_cache.py --mode train --num-samples 3
    python my_code/visualization/visualize_mo2cap2_v2_cache.py --mode test --num-samples 3
    python my_code/visualization/visualize_mo2cap2_v2_cache.py --mode both --num-samples 2
"""

import argparse
import os
import h5py
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mo2Cap2 constants
MO2CAP2_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # Right arm
    (0, 4), (4, 5), (5, 6),      # Left arm
    (0, 7), (7, 8), (8, 9), (9, 10),    # Right leg
    (0, 11), (11, 12), (12, 13), (13, 14),  # Left leg
]

MO2CAP2_JOINT_NAMES = [
    'Neck', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'
]

UPPER_BODY_IDX = list(range(7))  # Neck + arms
LOWER_BODY_IDX = list(range(7, 15))  # Legs


def load_train_sample(data_root, sample_idx):
    """Load a training sample from H5 chunks."""
    chunk_files = sorted([
        f for f in os.listdir(data_root)
        if f.startswith('mo2cap2_chunk_') and f.endswith('.hdf5')
    ])

    cumsum = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(data_root, chunk_file)
        with h5py.File(chunk_path, 'r') as hf:
            chunk_size = hf['Images'].shape[0]
            if cumsum + chunk_size > sample_idx:
                local_idx = sample_idx - cumsum
                image = hf['Images'][local_idx]  # (3, 256, 256)
                keypoint2d = hf['Annot2D'][local_idx].copy()  # (15, 2)
                keypoint3d = hf['Annot3D'][local_idx]  # (15, 3)

                # Apply X coordinate offset (-33) for Mo2Cap2 image alignment
                # This matches what the training pipeline does
                keypoint2d[:, 0] = keypoint2d[:, 0] - 33

                # CHW -> HWC, RGB
                image = np.transpose(image, (1, 2, 0))

                return {
                    'image': image,
                    'keypoint2d': keypoint2d,
                    'keypoint3d': keypoint3d,
                    'chunk_file': chunk_file,
                    'local_idx': local_idx,
                    'global_idx': sample_idx,
                    'has_2d': True
                }
            cumsum += chunk_size

    raise ValueError(f'Sample index {sample_idx} out of range (total: {cumsum})')


def load_test_sample(test_root, sequence='olek_outdoor', sample_idx=0):
    """Load a test sample from JPG + MAT files."""
    seq_dir = os.path.join(test_root, 'TestSet', sequence)
    mat_path = os.path.join(test_root, 'TestSet', f'{sequence}_gt.mat')

    # Load 3D GT
    mat = scipy.io.loadmat(mat_path)
    pose_gt = mat['pose_gt']  # (N, 15, 3)

    # Get image list
    img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])

    if sample_idx >= len(img_files):
        raise ValueError(f'Sample index {sample_idx} out of range (total: {len(img_files)})')

    img_path = os.path.join(seq_dir, img_files[sample_idx])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    keypoint3d = pose_gt[sample_idx]  # (15, 3)

    return {
        'image': image,
        'keypoint2d': None,  # No 2D GT in test set
        'keypoint3d': keypoint3d,
        'img_path': img_path,
        'sequence': sequence,
        'sample_idx': sample_idx,
        'has_2d': False
    }


def draw_2d_keypoints(ax, image, keypoint2d, title='2D View'):
    """Draw 2D image with keypoints."""
    ax.imshow(image)

    if keypoint2d is not None:
        # Draw skeleton
        for (i, j) in MO2CAP2_SKELETON:
            color = 'lime' if i in UPPER_BODY_IDX and j in UPPER_BODY_IDX else 'red'
            ax.plot([keypoint2d[i, 0], keypoint2d[j, 0]],
                    [keypoint2d[i, 1], keypoint2d[j, 1]],
                    c=color, linewidth=2)

        # Draw joints
        for i, (x, y) in enumerate(keypoint2d):
            color = 'lime' if i in UPPER_BODY_IDX else 'red'
            ax.scatter(x, y, c=color, s=40, zorder=5, edgecolors='white', linewidths=0.5)

    ax.set_title(title)
    ax.axis('off')


def draw_3d_pose(ax, keypoint3d, title='3D Joints', elev=15, azim=70):
    """Draw 3D skeleton."""
    # Make root-relative (Neck at origin)
    kp = keypoint3d - keypoint3d[0]

    # Draw skeleton
    for (i, j) in MO2CAP2_SKELETON:
        color = 'blue' if i in UPPER_BODY_IDX and j in UPPER_BODY_IDX else 'darkred'
        ax.plot([kp[i, 0], kp[j, 0]],
                [kp[i, 1], kp[j, 1]],
                [kp[i, 2], kp[j, 2]],
                c=color, linewidth=2)

    # Draw joints
    colors = ['blue' if i in UPPER_BODY_IDX else 'darkred' for i in range(15)]
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c=colors, s=50, marker='o')

    # Mark neck
    ax.scatter(0, 0, 0, c='cyan', s=100, marker='*', label='Neck (root)')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper right', fontsize=8)


def visualize_train_sample(sample, output_path=None):
    """Visualize training sample: 2D photo + 2D joints + 3D joints."""
    fig = plt.figure(figsize=(15, 5))

    # 1. 2D image with keypoints
    ax1 = fig.add_subplot(1, 3, 1)
    draw_2d_keypoints(ax1, sample['image'], sample['keypoint2d'],
                      title=f"Train Sample {sample['global_idx']}\n(2D Photo + 2D Joints)")

    # 2. 3D pose - front view
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_3d_pose(ax2, sample['keypoint3d'], title='3D Joints (Front View)', elev=15, azim=70)

    # 3. 3D pose - side view
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_3d_pose(ax3, sample['keypoint3d'], title='3D Joints (Side View)', elev=0, azim=0)

    # Info text
    info_text = (
        f"Training Data Structure:\n"
        f"  Image: {sample['image'].shape}\n"
        f"  2D Keypoints: {sample['keypoint2d'].shape}\n"
        f"  3D Keypoints: {sample['keypoint3d'].shape}\n"
        f"  Source: {sample['chunk_file']}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    plt.close()


def visualize_test_sample(sample, output_path=None):
    """Visualize test sample: 2D photo + 3D joints (no 2D GT)."""
    fig = plt.figure(figsize=(15, 5))

    # 1. 2D image only (no 2D keypoints in test set)
    ax1 = fig.add_subplot(1, 3, 1)
    draw_2d_keypoints(ax1, sample['image'], None,  # No 2D keypoints
                      title=f"Test Sample {sample['sample_idx']}\n(2D Photo Only - No 2D GT)")

    # 2. 3D pose - front view
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_3d_pose(ax2, sample['keypoint3d'], title='3D Joints (Front View)', elev=15, azim=70)

    # 3. 3D pose - side view
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_3d_pose(ax3, sample['keypoint3d'], title='3D Joints (Side View)', elev=0, azim=0)

    # Info text
    info_text = (
        f"Test Data Structure:\n"
        f"  Image: {sample['image'].shape}\n"
        f"  2D Keypoints: None (NOT available)\n"
        f"  3D Keypoints: {sample['keypoint3d'].shape}\n"
        f"  Sequence: {sample['sequence']}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Mo2Cap2 V2 Cache Dataset')
    parser.add_argument('--train-root', type=str,
                        default='/mnt/sdb2/mo2cap2_dataset/training_data',
                        help='Path to training H5 chunks')
    parser.add_argument('--test-root', type=str,
                        default='/mnt/sdb2/mo2cap2_dataset/test_data',
                        help='Path to test data')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                        help='Which dataset to visualize')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='output_v2_cache_vis',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('Mo2Cap2 V2 Cache Dataset Visualization')
    print('=' * 60)
    print()
    print('Dataset Structure:')
    print('  TRAIN: 2D photos + 2D joints + 3D joints (H5 chunks)')
    print('  TEST:  2D photos + 3D joints only (JPG + MAT files)')
    print()

    if args.mode in ['train', 'both']:
        print('-' * 60)
        print('Visualizing TRAINING samples...')
        print('-' * 60)

        np.random.seed(42)
        train_indices = np.random.randint(0, 50000, size=args.num_samples).tolist()

        for idx in train_indices:
            try:
                sample = load_train_sample(args.train_root, idx)
                output_path = os.path.join(args.output_dir, f'train_sample_{idx:06d}.png')
                visualize_train_sample(sample, output_path)

                print(f'  Sample {idx}:')
                print(f'    Image: {sample["image"].shape}')
                print(f'    2D Keypoints: {sample["keypoint2d"].shape} (X range: {sample["keypoint2d"][:, 0].min():.1f} ~ {sample["keypoint2d"][:, 0].max():.1f})')
                print(f'    3D Keypoints: {sample["keypoint3d"].shape}')
                print()
            except Exception as e:
                print(f'  Error loading sample {idx}: {e}')

    if args.mode in ['test', 'both']:
        print('-' * 60)
        print('Visualizing TEST samples...')
        print('-' * 60)

        for seq in ['olek_outdoor', 'weipeng_studio']:
            print(f'\n  Sequence: {seq}')

            np.random.seed(42)
            test_indices = np.random.randint(0, 1000, size=args.num_samples).tolist()

            for idx in test_indices:
                try:
                    sample = load_test_sample(args.test_root, sequence=seq, sample_idx=idx)
                    output_path = os.path.join(args.output_dir, f'test_{seq}_{idx:04d}.png')
                    visualize_test_sample(sample, output_path)

                    print(f'    Sample {idx}:')
                    print(f'      Image: {sample["image"].shape}')
                    print(f'      2D Keypoints: None (NOT available in test set)')
                    print(f'      3D Keypoints: {sample["keypoint3d"].shape}')
                except Exception as e:
                    print(f'    Error loading sample {idx}: {e}')

    print()
    print('=' * 60)
    print(f'Output saved to: {args.output_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
