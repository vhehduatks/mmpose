"""
Dataset Visualization Script for XR EgoPose

This script visualizes the dataset samples that are fed into the model,
including images, 2D/3D keypoints, heatmaps, and HMD info.

Usage:
    python my_code/visualize_dataset.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --num-samples 5 \
        --output output_dataset_vis/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Add mmpose to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS


# EgoPose skeleton (16 keypoints)
EGOPOSE_SKELETON = [
    [0, 1],   # Spine2 -> Head
    [0, 2],   # Spine2 -> LeftArm
    [2, 3],   # LeftArm -> LeftForeArm
    [3, 4],   # LeftForeArm -> LeftHand
    [0, 5],   # Spine2 -> RightArm
    [5, 6],   # RightArm -> RightForeArm
    [6, 7],   # RightForeArm -> RightHand
    [0, 8],   # Spine2 -> LeftUpLeg
    [8, 9],   # LeftUpLeg -> LeftLeg
    [9, 10],  # LeftLeg -> LeftFoot
    [10, 11], # LeftFoot -> LeftToeBase
    [0, 12],  # Spine2 -> RightUpLeg
    [12, 13], # RightUpLeg -> RightLeg
    [13, 14], # RightLeg -> RightFoot
    [14, 15], # RightFoot -> RightToeBase
]

# Keypoint colors
EGOPOSE_KPT_COLORS = np.array([
    [51, 153, 255],   # Spine2
    [51, 153, 255],   # Head
    [51, 153, 255],   # LeftArm
    [0, 255, 0],      # LeftForeArm
    [0, 255, 0],      # LeftHand
    [51, 153, 255],   # RightArm
    [255, 128, 0],    # RightForeArm
    [255, 128, 0],    # RightHand
    [51, 153, 255],   # LeftUpLeg
    [0, 255, 0],      # LeftLeg
    [0, 255, 0],      # LeftFoot
    [0, 255, 0],      # LeftToeBase
    [51, 153, 255],   # RightUpLeg
    [255, 128, 0],    # RightLeg
    [255, 128, 0],    # RightFoot
    [255, 128, 0],    # RightToeBase
]) / 255.0

# Link colors
EGOPOSE_LINK_COLORS = np.array([
    [51, 153, 255],   # Spine2 -> Head
    [51, 153, 255],   # Spine2 -> LeftArm
    [0, 255, 0],      # LeftArm -> LeftForeArm
    [0, 255, 0],      # LeftForeArm -> LeftHand
    [51, 153, 255],   # Spine2 -> RightArm
    [255, 128, 0],    # RightArm -> RightForeArm
    [255, 128, 0],    # RightForeArm -> RightHand
    [51, 153, 255],   # Spine2 -> LeftUpLeg
    [0, 255, 0],      # LeftUpLeg -> LeftLeg
    [0, 255, 0],      # LeftLeg -> LeftFoot
    [0, 255, 0],      # LeftFoot -> LeftToeBase
    [51, 153, 255],   # Spine2 -> RightUpLeg
    [255, 128, 0],    # RightUpLeg -> RightLeg
    [255, 128, 0],    # RightLeg -> RightFoot
    [255, 128, 0],    # RightFoot -> RightToeBase
]) / 255.0

# Keypoint names
EGOPOSE_KPT_NAMES = [
    'Spine2', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize XR EgoPose Dataset')
    parser.add_argument('--config', type=str,
                        default='my_code/custom_config/HMD_xregopose_h5cache_config.py',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='output_dataset_vis',
                        help='Output directory for visualization')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization window')
    parser.add_argument('--random', action='store_true',
                        help='Randomly sample from dataset')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info about sample structure')
    return parser.parse_args()


def draw_2d_skeleton(ax, image, keypoints, title='2D Keypoints'):
    """Draw 2D skeleton on image.

    Args:
        ax: Matplotlib axis
        image: RGB image (H, W, 3)
        keypoints: (N, 2) or (1, N, 2) array of 2D keypoints
        title: Plot title
    """
    ax.imshow(image)

    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    h, w = image.shape[:2]

    # Draw skeleton lines first
    for idx, (i, j) in enumerate(EGOPOSE_SKELETON):
        if i < len(keypoints) and j < len(keypoints):
            pt1, pt2 = keypoints[i], keypoints[j]
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color=EGOPOSE_LINK_COLORS[idx], linewidth=2)

    # Draw keypoints
    for i, kpt in enumerate(keypoints):
        if i < len(EGOPOSE_KPT_COLORS):
            if 0 <= kpt[0] < w and 0 <= kpt[1] < h:
                ax.scatter(kpt[0], kpt[1], c=[EGOPOSE_KPT_COLORS[i]],
                          s=50, marker='o', edgecolors='white', linewidths=1, zorder=5)

    ax.set_title(title)
    ax.axis('off')


def draw_3d_skeleton(ax, keypoints, title='3D Keypoints'):
    """Draw 3D skeleton.

    Args:
        ax: Matplotlib 3D axis
        keypoints: (N, 3) or (1, N, 3) array of 3D keypoints
        title: Plot title
    """
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    # Draw keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
               c=EGOPOSE_KPT_COLORS, s=50, marker='o')

    # Draw skeleton
    for idx, (i, j) in enumerate(EGOPOSE_SKELETON):
        if i < len(keypoints) and j < len(keypoints):
            pts = keypoints[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=EGOPOSE_LINK_COLORS[idx], linewidth=2)

    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio
    center = keypoints.mean(axis=0)
    max_range = np.abs(keypoints - center).max() * 1.2
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    ax.view_init(elev=15, azim=70)


def draw_heatmaps(ax, heatmaps, title='Heatmaps (Sum)'):
    """Draw sum of all heatmaps.

    Args:
        ax: Matplotlib axis
        heatmaps: (C, H, W) array of heatmaps
        title: Plot title
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.numpy()

    # Sum all channels
    heatmap_sum = heatmaps.sum(axis=0)

    ax.imshow(heatmap_sum, cmap='jet')
    ax.set_title(title)
    ax.axis('off')


def draw_individual_heatmaps(fig, heatmaps, start_row, num_cols=8):
    """Draw individual heatmaps in a grid.

    Args:
        fig: Matplotlib figure
        heatmaps: (C, H, W) array of heatmaps
        start_row: Starting row in the grid
        num_cols: Number of columns in grid
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.numpy()

    num_kpts = min(heatmaps.shape[0], 16)
    num_rows = (num_kpts + num_cols - 1) // num_cols

    for i in range(num_kpts):
        ax = fig.add_subplot(4 + num_rows, num_cols, start_row * num_cols + i + 1)
        ax.imshow(heatmaps[i], cmap='jet')
        ax.set_title(EGOPOSE_KPT_NAMES[i], fontsize=8)
        ax.axis('off')


def visualize_hmd_info(ax, hmd_info, title='HMD Info'):
    """Visualize HMD info as a 3D plot.

    Args:
        ax: Matplotlib 3D axis
        hmd_info: (9,) or (1, 9) array of HMD direction vectors
        title: Plot title
    """
    if hmd_info.ndim == 2:
        hmd_info = hmd_info[0]

    # HMD info contains: head_dir(3), left_hand_dir(3), right_hand_dir(3)
    origin = np.array([0, 0, 0])
    head_dir = hmd_info[0:3]
    left_hand_dir = hmd_info[3:6]
    right_hand_dir = hmd_info[6:9]

    # Draw direction vectors as arrows
    ax.quiver(*origin, *head_dir, color='blue', arrow_length_ratio=0.1, label='Head')
    ax.quiver(*origin, *left_hand_dir, color='green', arrow_length_ratio=0.1, label='Left Hand')
    ax.quiver(*origin, *right_hand_dir, color='orange', arrow_length_ratio=0.1, label='Right Hand')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    ax.view_init(elev=20, azim=45)


def print_sample_structure(sample, prefix='', depth=0):
    """Print the structure of a sample for debugging."""
    if depth > 3:  # Limit recursion depth
        return
    if isinstance(sample, dict):
        for k, v in sample.items():
            print(f"{prefix}{k}: {type(v).__name__}")
            if isinstance(v, (dict, )):
                print_sample_structure(v, prefix + '  ', depth + 1)
            elif isinstance(v, torch.Tensor):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}, min={v.min().item():.3f}, max={v.max().item():.3f}")
            elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
            elif isinstance(v, np.ndarray):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, '__dict__'):
                print_sample_structure(v, prefix + '  ', depth + 1)
    elif hasattr(sample, '__dict__'):
        for k, v in vars(sample).items():
            if k.startswith('_'):
                continue
            print(f"{prefix}{k}: {type(v).__name__}")
            if isinstance(v, torch.Tensor):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}, min={v.min().item():.3f}, max={v.max().item():.3f}")
            elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
            elif isinstance(v, np.ndarray):
                print(f"{prefix}  shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, '__dict__') and not k.startswith('_'):
                print_sample_structure(v, prefix + '  ', depth + 1)


def visualize_sample(sample, output_path=None, show=False, debug=False):
    """Visualize a single dataset sample.

    Args:
        sample: Dataset sample dictionary
        output_path: Path to save visualization
        show: Whether to show the plot
        debug: Whether to print debug info
    """
    if debug:
        print("\n=== Sample Structure ===")
        print_sample_structure(sample)
        print("========================\n")

    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data from sample
    data_samples = sample.get('data_samples', None)
    inputs = sample.get('inputs', None)

    # Get image - dataset returns image in [0, 255] range, RGB format, (C, H, W)
    if inputs is not None:
        if isinstance(inputs, torch.Tensor):
            # (C, H, W) -> (H, W, C)
            img = inputs.permute(1, 2, 0).numpy()
            # Image from dataset is NOT normalized (data_preprocessor does that during training)
            # Just clip and convert to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.array(inputs)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
    else:
        img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Get ground truth data
    gt_heatmaps = None
    gt_keypoints_2d = None
    gt_keypoints_3d = None
    hmd_info = None
    img_path = None
    transformed_keypoints = None

    if data_samples is not None:
        # Get image path
        if hasattr(data_samples, 'img_path'):
            img_path = data_samples.img_path

        # Get GT fields
        if hasattr(data_samples, 'gt_fields'):
            gt_fields = data_samples.gt_fields
            if 'heatmaps' in gt_fields:
                gt_heatmaps = gt_fields.heatmaps

        if hasattr(data_samples, 'gt_instances'):
            gt_inst = data_samples.gt_instances
            # Get transformed keypoints (in the cropped image coordinate)
            if 'transformed_keypoints' in gt_inst:
                transformed_keypoints = gt_inst.transformed_keypoints
            if 'keypoints' in gt_inst:
                gt_keypoints_2d = gt_inst.keypoints
            if 'keypoint3d' in gt_inst:
                gt_keypoints_3d = gt_inst.keypoint3d
            elif 'lifting_target' in gt_inst:
                gt_keypoints_3d = gt_inst.lifting_target

        if hasattr(data_samples, 'gt_instance_labels'):
            gt_labels = data_samples.gt_instance_labels
            if 'hmd_info' in gt_labels:
                hmd_info = gt_labels.hmd_info

    # Use transformed_keypoints for visualization if available (they match the cropped image)
    if transformed_keypoints is not None:
        gt_keypoints_2d = transformed_keypoints

    # Convert tensors to numpy
    if isinstance(gt_keypoints_2d, torch.Tensor):
        gt_keypoints_2d = gt_keypoints_2d.numpy()
    if isinstance(gt_keypoints_3d, torch.Tensor):
        gt_keypoints_3d = gt_keypoints_3d.numpy()
    if isinstance(hmd_info, torch.Tensor):
        hmd_info = hmd_info.numpy()

    # Row 1: Input image and 2D keypoints
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Input Image (Preprocessed)')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    if gt_keypoints_2d is not None:
        draw_2d_skeleton(ax2, img, gt_keypoints_2d, 'GT 2D Keypoints')
    else:
        ax2.imshow(img)
        ax2.set_title('No 2D Keypoints')
        ax2.axis('off')

    # Row 1: 3D keypoints and HMD info
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    if gt_keypoints_3d is not None:
        draw_3d_skeleton(ax3, gt_keypoints_3d, 'GT 3D Keypoints')
    else:
        ax3.set_title('No 3D Keypoints')

    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    if hmd_info is not None:
        visualize_hmd_info(ax4, hmd_info, 'HMD Direction Vectors')
    else:
        ax4.set_title('No HMD Info')

    # Row 2: Heatmaps
    if gt_heatmaps is not None:
        if isinstance(gt_heatmaps, torch.Tensor):
            gt_heatmaps = gt_heatmaps.numpy()

        # Sum of heatmaps
        ax5 = fig.add_subplot(gs[1, 0])
        draw_heatmaps(ax5, gt_heatmaps, 'Heatmaps (Sum)')

        # Individual heatmaps - show first 12 in remaining space
        num_show = min(12, gt_heatmaps.shape[0])
        for i in range(num_show):
            row = 1 + (i // 4)
            col = i % 4 if row > 1 else (i % 3) + 1
            if row == 1:
                ax = fig.add_subplot(gs[1, col])
            else:
                ax = fig.add_subplot(gs[row, i % 4])
            ax.imshow(gt_heatmaps[i], cmap='jet')
            ax.set_title(f'{EGOPOSE_KPT_NAMES[i]}', fontsize=9)
            ax.axis('off')

    # Add image path as suptitle
    if img_path:
        fig.suptitle(f'Image: {img_path}', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load config
    cfg = Config.fromfile(args.config)

    # Initialize mmpose scope
    init_default_scope('mmpose')

    # Get dataset config based on split
    if args.split == 'train':
        dataset_cfg = cfg.train_dataloader.dataset
    elif args.split == 'val':
        dataset_cfg = cfg.val_dataloader.dataset
    else:
        dataset_cfg = cfg.test_dataloader.dataset

    # Build dataset
    print(f"Building {args.split} dataset...")
    dataset = DATASETS.build(dataset_cfg)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Get sample indices
    if args.random:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    else:
        indices = range(min(args.num_samples, len(dataset)))

    # Visualize samples
    for i, idx in enumerate(indices):
        print(f"Visualizing sample {i+1}/{len(indices)} (index: {idx})")

        sample = dataset[idx]
        output_path = os.path.join(args.output, f'{args.split}_sample_{idx:05d}.png')
        # Only print debug for first sample
        visualize_sample(sample, output_path, args.show, debug=(args.debug and i == 0))

    print(f"\nDone! Visualizations saved to {args.output}")


if __name__ == '__main__':
    main()
