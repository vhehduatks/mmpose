"""
Skeleton Labeling Check Script for XR EgoPose Dataset

This script visualizes the skeleton with joint indices and names to verify
that the skeleton labeling is correct.

Usage:
    # Visualize from H5 cache dataset
    python my_code/visualization/check_skeleton_labeling.py \
        --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
        --num-samples 5 \
        --output output_skeleton_check/

    # Visualize specific sample index
    python my_code/visualization/check_skeleton_labeling.py \
        --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
        --sample-idx 100 \
        --output output_skeleton_check/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add mmpose to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Skeleton Definition
# =============================================================================
# IMPORTANT: There are TWO different skeleton orderings in this codebase:
#
# 1. config.py (used for data loading from JSON → H5 cache):
#    0:Head, 1:Neck, 2:LeftArm, 3:LeftForeArm, 4:LeftHand,
#    5:RightArm, 6:RightForeArm, 7:RightHand, 8:LeftUpLeg, 9:LeftLeg,
#    10:LeftFoot, 11:LeftToeBase, 12:RightUpLeg, 13:RightLeg, 14:RightFoot, 15:RightToeBase
#
# 2. egopose_info.py (used for mmpose visualization/metrics):
#    0:Spine2, 1:Head, 2:LeftArm, 3:LeftForeArm, 4:LeftHand,
#    5:RightArm, 6:RightForeArm, 7:RightHand, 8:LeftUpLeg, 9:LeftLeg,
#    10:LeftFoot, 11:LeftToeBase, 12:RightUpLeg, 13:RightLeg, 14:RightFoot, 15:RightToeBase
#
# The H5 cache uses config.py ordering, so we use that here.
# =============================================================================

# Joint names from config.py (actual data order)
JOINT_NAMES_CONFIG = [
    'Head',         # 0
    'Neck',         # 1
    'LeftArm',      # 2  (LeftShoulder)
    'LeftForeArm',  # 3  (LeftElbow)
    'LeftHand',     # 4
    'RightArm',     # 5  (RightShoulder)
    'RightForeArm', # 6  (RightElbow)
    'RightHand',    # 7
    'LeftUpLeg',    # 8  (LeftHip)
    'LeftLeg',      # 9  (LeftKnee)
    'LeftFoot',     # 10 (LeftAnkle)
    'LeftToeBase',  # 11
    'RightUpLeg',   # 12 (RightHip)
    'RightLeg',     # 13 (RightKnee)
    'RightFoot',    # 14 (RightAnkle)
    'RightToeBase', # 15
]

# Joint names from egopose_info.py (mmpose visualization order)
JOINT_NAMES_EGOPOSE_INFO = [
    'Spine2',       # 0  <- Different!
    'Head',         # 1  <- Different!
    'LeftArm',      # 2
    'LeftForeArm',  # 3
    'LeftHand',     # 4
    'RightArm',     # 5
    'RightForeArm', # 6
    'RightHand',    # 7
    'LeftUpLeg',    # 8
    'LeftLeg',      # 9
    'LeftFoot',     # 10
    'LeftToeBase',  # 11
    'RightUpLeg',   # 12
    'RightLeg',     # 13
    'RightFoot',    # 14
    'RightToeBase', # 15
]

# Default: use config.py ordering (matches H5 cache data)
JOINT_NAMES = JOINT_NAMES_CONFIG

# Skeleton connections based on config.py ordering
# Since Spine2 is not in the skeleton, we connect through Neck
SKELETON_LINKS = [
    # Head chain
    (0, 1),    # Head -> Neck
    # Left arm chain (from Neck since no Spine2)
    (1, 2),    # Neck -> LeftArm
    (2, 3),    # LeftArm -> LeftForeArm
    (3, 4),    # LeftForeArm -> LeftHand
    # Right arm chain
    (1, 5),    # Neck -> RightArm
    (5, 6),    # RightArm -> RightForeArm
    (6, 7),    # RightForeArm -> RightHand
    # Left leg chain (via Neck since no Hips/Spine in skeleton)
    (1, 8),    # Neck -> LeftUpLeg (simplified - missing Spine/Hips)
    (8, 9),    # LeftUpLeg -> LeftLeg
    (9, 10),   # LeftLeg -> LeftFoot
    (10, 11),  # LeftFoot -> LeftToeBase
    # Right leg chain
    (1, 12),   # Neck -> RightUpLeg (simplified)
    (12, 13),  # RightUpLeg -> RightLeg
    (13, 14),  # RightLeg -> RightFoot
    (14, 15),  # RightFoot -> RightToeBase
]

# Colors for different body parts
COLORS = {
    'head': '#3399FF',      # Blue - Head/Neck
    'left_arm': '#00FF00',  # Green - Left arm
    'right_arm': '#FF8000', # Orange - Right arm
    'left_leg': '#00FFFF',  # Cyan - Left leg
    'right_leg': '#FF00FF', # Magenta - Right leg
}

# Joint to color mapping
JOINT_COLORS = [
    COLORS['head'],      # 0: Head
    COLORS['head'],      # 1: Neck
    COLORS['left_arm'],  # 2: LeftArm
    COLORS['left_arm'],  # 3: LeftForeArm
    COLORS['left_arm'],  # 4: LeftHand
    COLORS['right_arm'], # 5: RightArm
    COLORS['right_arm'], # 6: RightForeArm
    COLORS['right_arm'], # 7: RightHand
    COLORS['left_leg'],  # 8: LeftUpLeg
    COLORS['left_leg'],  # 9: LeftLeg
    COLORS['left_leg'],  # 10: LeftFoot
    COLORS['left_leg'],  # 11: LeftToeBase
    COLORS['right_leg'], # 12: RightUpLeg
    COLORS['right_leg'], # 13: RightLeg
    COLORS['right_leg'], # 14: RightFoot
    COLORS['right_leg'], # 15: RightToeBase
]

# Link colors based on body part
LINK_COLORS = [
    COLORS['head'],      # Head -> Neck
    COLORS['left_arm'],  # Neck -> LeftArm
    COLORS['left_arm'],  # LeftArm -> LeftForeArm
    COLORS['left_arm'],  # LeftForeArm -> LeftHand
    COLORS['right_arm'], # Neck -> RightArm
    COLORS['right_arm'], # RightArm -> RightForeArm
    COLORS['right_arm'], # RightForeArm -> RightHand
    COLORS['left_leg'],  # Neck -> LeftUpLeg
    COLORS['left_leg'],  # LeftUpLeg -> LeftLeg
    COLORS['left_leg'],  # LeftLeg -> LeftFoot
    COLORS['left_leg'],  # LeftFoot -> LeftToeBase
    COLORS['right_leg'], # Neck -> RightUpLeg
    COLORS['right_leg'], # RightUpLeg -> RightLeg
    COLORS['right_leg'], # RightLeg -> RightFoot
    COLORS['right_leg'], # RightFoot -> RightToeBase
]


def parse_args():
    parser = argparse.ArgumentParser(description='Check Skeleton Labeling')
    parser.add_argument('--cache-file', type=str, required=True,
                        help='Path to H5 cache file')
    parser.add_argument('--sample-idx', type=int, default=None,
                        help='Specific sample index to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='output_skeleton_check',
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization window')
    parser.add_argument('--random', action='store_true',
                        help='Random sample selection')
    parser.add_argument('--use-egopose-info-order', action='store_true',
                        help='Use egopose_info.py ordering instead of config.py')
    return parser.parse_args()


def set_joint_names(use_egopose_info_order=False):
    """Set global JOINT_NAMES based on selected ordering."""
    global JOINT_NAMES
    if use_egopose_info_order:
        JOINT_NAMES = JOINT_NAMES_EGOPOSE_INFO
        print("Using egopose_info.py joint ordering (0=Spine2)")
    else:
        JOINT_NAMES = JOINT_NAMES_CONFIG
        print("Using config.py joint ordering (0=Head)")


def draw_skeleton_2d(ax, image, keypoints_2d, show_labels=True, title='2D Skeleton'):
    """Draw 2D skeleton with joint labels on image.

    Args:
        ax: Matplotlib axis
        image: RGB image (H, W, 3)
        keypoints_2d: (16, 2) array of 2D keypoints
        show_labels: Whether to show joint index and name labels
        title: Plot title
    """
    ax.imshow(image)

    if keypoints_2d.ndim == 3:
        keypoints_2d = keypoints_2d[0]

    h, w = image.shape[:2]

    # Draw skeleton links first
    for idx, (i, j) in enumerate(SKELETON_LINKS):
        if i < len(keypoints_2d) and j < len(keypoints_2d):
            pt1 = keypoints_2d[i]
            pt2 = keypoints_2d[j]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    color=LINK_COLORS[idx], linewidth=2, alpha=0.8)

    # Draw keypoints with labels
    for i, kpt in enumerate(keypoints_2d):
        color = JOINT_COLORS[i]

        # Draw keypoint
        ax.scatter(kpt[0], kpt[1], c=color, s=100, marker='o',
                   edgecolors='white', linewidths=2, zorder=10)

        # Draw label
        if show_labels:
            label = f"{i}: {JOINT_NAMES[i]}"
            # Offset label to avoid overlapping with point
            offset_x = 5
            offset_y = -10 if i % 2 == 0 else 10
            ax.annotate(label, (kpt[0], kpt[1]),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.8))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def draw_skeleton_3d(ax, keypoints_3d, show_labels=True, title='3D Skeleton',
                     elev=20, azim=45):
    """Draw 3D skeleton with joint labels.

    Args:
        ax: Matplotlib 3D axis
        keypoints_3d: (16, 3) array of 3D keypoints
        show_labels: Whether to show joint index and name labels
        title: Plot title
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    if keypoints_3d.ndim == 3:
        keypoints_3d = keypoints_3d[0]

    # Draw skeleton links
    for idx, (i, j) in enumerate(SKELETON_LINKS):
        if i < len(keypoints_3d) and j < len(keypoints_3d):
            pts = keypoints_3d[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=LINK_COLORS[idx], linewidth=2, alpha=0.8)

    # Draw keypoints
    for i, kpt in enumerate(keypoints_3d):
        color = JOINT_COLORS[i]
        ax.scatter(kpt[0], kpt[1], kpt[2], c=color, s=80, marker='o',
                   edgecolors='white', linewidths=1, zorder=10)

        # Draw label
        if show_labels:
            label = f"{i}"  # Just index for 3D to avoid clutter
            ax.text(kpt[0], kpt[1], kpt[2], label, fontsize=8,
                    fontweight='bold', color='black',
                    ha='center', va='bottom')

    # Set axis properties
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Set equal aspect ratio
    center = keypoints_3d.mean(axis=0)
    max_range = np.abs(keypoints_3d - center).max() * 1.3
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    ax.view_init(elev=elev, azim=azim)


def draw_joint_legend(ax):
    """Draw legend showing joint names and colors."""
    ax.axis('off')
    ax.set_title('Joint Index Reference', fontsize=12, fontweight='bold')

    # Create text for each joint
    y_positions = np.linspace(0.95, 0.05, 16)
    for i, (name, color, y) in enumerate(zip(JOINT_NAMES, JOINT_COLORS, y_positions)):
        ax.text(0.1, y, f"{i:2d}: {name}", fontsize=10, fontweight='bold',
                color=color, transform=ax.transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9))


def visualize_sample(image, keypoints_2d, keypoints_3d, output_path=None,
                     show=False, sample_info=''):
    """Visualize a single sample with 2D and 3D skeletons.

    Args:
        image: RGB image (H, W, 3) or None
        keypoints_2d: (16, 2) array of 2D keypoints
        keypoints_3d: (16, 3) array of 3D keypoints
        output_path: Path to save visualization
        show: Whether to show the plot
        sample_info: Additional info to show in title
    """
    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Layout: 2 rows, 3 columns
    # Row 1: 2D skeleton on image, 3D view 1, 3D view 2
    # Row 2: Joint legend, 3D view 3 (from above), 3D view 4 (from front)

    # Ensure correct shapes
    if keypoints_2d.ndim == 3:
        keypoints_2d = keypoints_2d[0]
    if keypoints_3d.ndim == 3:
        keypoints_3d = keypoints_3d[0]

    # Row 1, Col 1: 2D skeleton on image
    ax1 = fig.add_subplot(2, 3, 1)
    if image is not None:
        draw_skeleton_2d(ax1, image, keypoints_2d, show_labels=True,
                         title='2D Keypoints (with labels)')
    else:
        ax1.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
        ax1.set_title('2D Keypoints')
        ax1.axis('off')

    # Row 1, Col 2: 3D skeleton (default view)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    draw_skeleton_3d(ax2, keypoints_3d, show_labels=True,
                     title='3D Skeleton (Side View)', elev=15, azim=45)

    # Row 1, Col 3: 3D skeleton (front view)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    draw_skeleton_3d(ax3, keypoints_3d, show_labels=True,
                     title='3D Skeleton (Front View)', elev=0, azim=0)

    # Row 2, Col 1: Joint legend
    ax4 = fig.add_subplot(2, 3, 4)
    draw_joint_legend(ax4)

    # Row 2, Col 2: 3D skeleton (top view)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    draw_skeleton_3d(ax5, keypoints_3d, show_labels=True,
                     title='3D Skeleton (Top View)', elev=90, azim=0)

    # Row 2, Col 3: 3D skeleton (back view)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    draw_skeleton_3d(ax6, keypoints_3d, show_labels=True,
                     title='3D Skeleton (Back View)', elev=15, azim=-135)

    # Add sample info as suptitle
    if sample_info:
        fig.suptitle(sample_info, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close()


def load_sample_from_h5(h5_file, idx):
    """Load a sample from H5 cache file.

    Args:
        h5_file: Open H5 file handle
        idx: Sample index

    Returns:
        dict with image, keypoints_2d, keypoints_3d, img_path
    """
    # Load image if available
    image = None
    if 'images' in h5_file:
        img_data = h5_file['images'][idx]
        if img_data.ndim == 3:
            image = img_data  # Already (H, W, C)
        elif img_data.ndim == 2:
            image = np.stack([img_data] * 3, axis=-1)  # Grayscale to RGB

    # Load keypoints
    keypoints_2d = h5_file['keypoints'][idx]  # (1, 16, 2)
    keypoints_3d = h5_file['keypoint3d'][idx]  # (1, 16, 3)

    # Load image path
    img_path = ''
    if 'img_paths' in h5_file:
        img_path = h5_file['img_paths'][idx]
        if isinstance(img_path, bytes):
            img_path = img_path.decode('utf-8')

    return {
        'image': image,
        'keypoints_2d': keypoints_2d,
        'keypoints_3d': keypoints_3d,
        'img_path': img_path,
    }


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print skeleton information - compare both orderings
    print("=" * 70)
    print("SKELETON JOINT REFERENCE COMPARISON")
    print("=" * 70)
    print(f"{'Index':<6} {'config.py (H5 data)':<20} {'egopose_info.py (vis)':<20}")
    print("-" * 70)
    for i in range(16):
        config_name = JOINT_NAMES_CONFIG[i]
        egopose_name = JOINT_NAMES_EGOPOSE_INFO[i]
        diff = " <-- DIFFERENT!" if config_name != egopose_name else ""
        print(f"{i:<6} {config_name:<20} {egopose_name:<20}{diff}")
    print("=" * 70)
    print("\nNOTE: H5 cache uses config.py ordering by default.")
    print("      Use --use-egopose-info-order to try egopose_info.py ordering.")
    print()

    # Set joint names based on selected ordering
    set_joint_names(args.use_egopose_info_order)

    # Open H5 cache file
    print(f"Loading from: {args.cache_file}")
    with h5py.File(args.cache_file, 'r') as hf:
        total_samples = len(hf['keypoints'])
        print(f"Total samples: {total_samples}")

        # Determine sample indices
        if args.sample_idx is not None:
            indices = [args.sample_idx]
        elif args.random:
            indices = np.random.choice(total_samples,
                                        min(args.num_samples, total_samples),
                                        replace=False)
        else:
            indices = range(min(args.num_samples, total_samples))

        # Visualize samples
        for i, idx in enumerate(indices):
            print(f"Visualizing sample {i+1}/{len(indices)} (index: {idx})")

            sample = load_sample_from_h5(hf, idx)

            # Prepare sample info
            sample_info = f"Sample Index: {idx}"
            if sample['img_path']:
                sample_info += f" | Path: {Path(sample['img_path']).name}"

            # Save visualization
            output_path = os.path.join(args.output, f'skeleton_check_{idx:05d}.png')
            visualize_sample(
                image=sample['image'],
                keypoints_2d=sample['keypoints_2d'],
                keypoints_3d=sample['keypoints_3d'],
                output_path=output_path,
                show=args.show,
                sample_info=sample_info
            )

    print(f"\nDone! Visualizations saved to {args.output}")
    print("\nTo verify skeleton labeling:")
    print("  - Check that joint indices match the expected body parts")
    print("  - Green = Left side, Orange/Magenta = Right side")
    print("  - Head/Neck should be at the top")
    print("  - Feet/Toes should be at the bottom")


if __name__ == '__main__':
    main()
