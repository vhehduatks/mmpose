"""
Mo2Cap2 Dataset Visualization

Visualize images, 2D/3D annotations, and heatmaps from the Mo2Cap2 dataset.

Usage:
    python my_code/visualization/visualize_mo2cap2.py --chunk 1 --sample 0
    python my_code/visualization/visualize_mo2cap2.py --chunk 1 --sample 0 --save
    python my_code/visualization/visualize_mo2cap2.py --test olek_outdoor --sample 0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import scipy.io as sio
from PIL import Image

# Mo2Cap2 skeleton definition (15 joints)
# Based on configs/_base_/datasets/custom_mo2cap2.py
MO2CAP2_JOINTS = [
    'Neck',           # 0
    'RightArm',       # 1
    'RightForeArm',   # 2
    'RightHand',      # 3
    'LeftArm',        # 4
    'LeftForeArm',    # 5
    'LeftHand',       # 6
    'RightUpLeg',     # 7
    'RightLeg',       # 8
    'RightFoot',      # 9
    'RightToeBase',   # 10
    'LeftUpLeg',      # 11
    'LeftLeg',        # 12
    'LeftFoot',       # 13
    'LeftToeBase',    # 14
]

# Skeleton connections (parent, child)
# Note: Arms connect directly to UpLegs (no explicit Hip/Torso joint)
MO2CAP2_SKELETON = [
    (0, 4),    # Neck -> LeftArm
    (0, 1),    # Neck -> RightArm
    (4, 5),    # LeftArm -> LeftForeArm
    (5, 6),    # LeftForeArm -> LeftHand
    (1, 2),    # RightArm -> RightForeArm
    (2, 3),    # RightForeArm -> RightHand
    (4, 11),   # LeftArm -> LeftUpLeg (torso connection)
    (11, 12),  # LeftUpLeg -> LeftLeg
    (12, 13),  # LeftLeg -> LeftFoot
    (13, 14),  # LeftFoot -> LeftToeBase
    (1, 7),    # RightArm -> RightUpLeg (torso connection)
    (7, 8),    # RightUpLeg -> RightLeg
    (8, 9),    # RightLeg -> RightFoot
    (9, 10),   # RightFoot -> RightToeBase
]

# Colors for left/right sides
COLORS = {
    'left': '#3498db',   # Blue
    'right': '#e74c3c',  # Red
    'center': '#2ecc71', # Green
}

def get_bone_color(parent, child):
    """Get color based on bone side."""
    # Updated for correct Mo2Cap2 joint indices
    left_joints = {4, 5, 6, 11, 12, 13, 14}   # LeftArm, LeftForeArm, LeftHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase
    right_joints = {1, 2, 3, 7, 8, 9, 10}     # RightArm, RightForeArm, RightHand, RightUpLeg, RightLeg, RightFoot, RightToeBase

    if parent in left_joints or child in left_joints:
        return COLORS['left']
    elif parent in right_joints or child in right_joints:
        return COLORS['right']
    else:
        return COLORS['center']


def load_training_sample(chunk_id, sample_id, data_root='/mnt/sdb2/mo2cap2_dataset'):
    """Load a sample from training HDF5 chunk.

    Preprocessing (same as mo2cap2_coco_dataset.py):
    1. 2D coordinates: X -= 33 (image offset correction)
    2. 3D coordinates: Subtract Neck position to make it (0,0,0) reference
    """
    chunk_path = os.path.join(data_root, 'training_data', f'mo2cap2_chunk_{chunk_id:04d}.hdf5')

    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")

    with h5py.File(chunk_path, 'r') as f:
        # Images are stored as (N, C, H, W), need to transpose to (H, W, C)
        image = f['Images'][sample_id].transpose(1, 2, 0)
        zoom_image = f['ZoomImages'][sample_id].transpose(1, 2, 0)
        annot_2d = f['Annot2D'][sample_id].copy()  # (15, 2)
        annot_3d = f['Annot3D'][sample_id].copy()  # (15, 3)
        heatmaps = f['Heatmaps'][sample_id]  # (15, 32, 32)
        zoom_heatmaps = f['ZoomHeatmaps'][sample_id]  # (15, 32, 32)

    # Preprocessing Step 1: Subtract 33 from 2D X coordinate (image offset)
    annot_2d[:, 0] -= 33

    # Preprocessing Step 2: Set Neck (index 0) as 3D reference point (0,0,0)
    annot_3d -= annot_3d[0]

    return {
        'image': image,
        'zoom_image': zoom_image,
        'annot_2d': annot_2d,
        'annot_3d': annot_3d,
        'heatmaps': heatmaps,
        'zoom_heatmaps': zoom_heatmaps,
        'chunk_id': chunk_id,
        'sample_id': sample_id,
    }


def load_test_sample(test_set, sample_id, data_root='/mnt/sdb2/mo2cap2_dataset'):
    """Load a sample from test set."""
    test_dir = os.path.join(data_root, 'test_data', 'TestSet', test_set)
    gt_path = os.path.join(data_root, 'test_data', 'TestSet', f'{test_set}_gt.mat')

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test set not found: {test_dir}")

    # Get image list
    images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    if sample_id >= len(images):
        raise IndexError(f"Sample {sample_id} out of range (max: {len(images)-1})")

    # Load image
    img_path = os.path.join(test_dir, images[sample_id])
    image = np.array(Image.open(img_path))

    # Load ground truth
    gt = sio.loadmat(gt_path)
    pose_gt = gt['pose_gt']  # (N, 15, 3)

    # Note: GT may have more samples than images due to frame skipping
    annot_3d = pose_gt[sample_id] if sample_id < len(pose_gt) else None

    return {
        'image': image,
        'annot_3d': annot_3d,
        'test_set': test_set,
        'sample_id': sample_id,
        'img_name': images[sample_id],
    }


def visualize_2d(ax, image, annot_2d, title='2D Keypoints'):
    """Visualize 2D keypoints on image."""
    ax.imshow(image)
    ax.set_title(title)

    if annot_2d is not None:
        # Draw skeleton
        for parent, child in MO2CAP2_SKELETON:
            color = get_bone_color(parent, child)
            ax.plot(
                [annot_2d[parent, 0], annot_2d[child, 0]],
                [annot_2d[parent, 1], annot_2d[child, 1]],
                color=color, linewidth=2, alpha=0.8
            )

        # Draw joints
        for i, (x, y) in enumerate(annot_2d):
            # Updated for correct Mo2Cap2 joint indices
            if i in {4, 5, 6, 11, 12, 13, 14}:  # Left side
                color = COLORS['left']
            elif i in {1, 2, 3, 7, 8, 9, 10}:   # Right side
                color = COLORS['right']
            else:  # Neck (0)
                color = COLORS['center']

            ax.scatter(x, y, c=color, s=50, zorder=5, edgecolors='white', linewidths=1)
            ax.annotate(str(i), (x+3, y+3), fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

    ax.axis('off')


def visualize_3d(ax, annot_3d, title='3D Pose', elev=20, azim=45):
    """Visualize 3D pose."""
    ax.set_title(title)

    if annot_3d is None:
        ax.text(0.5, 0.5, 0.5, 'No 3D annotation', ha='center', va='center')
        return

    # Draw skeleton
    for parent, child in MO2CAP2_SKELETON:
        color = get_bone_color(parent, child)
        ax.plot3D(
            [annot_3d[parent, 0], annot_3d[child, 0]],
            [annot_3d[parent, 1], annot_3d[child, 1]],
            [annot_3d[parent, 2], annot_3d[child, 2]],
            color=color, linewidth=2
        )

    # Draw joints
    for i, (x, y, z) in enumerate(annot_3d):
        # Updated for correct Mo2Cap2 joint indices
        if i in {4, 5, 6, 11, 12, 13, 14}:  # Left side
            color = COLORS['left']
        elif i in {1, 2, 3, 7, 8, 9, 10}:   # Right side
            color = COLORS['right']
        else:  # Neck (0)
            color = COLORS['center']
        ax.scatter3D(x, y, z, c=color, s=50, edgecolors='white', linewidths=1)

    # Set equal aspect ratio
    max_range = np.max(np.abs(annot_3d)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)


def visualize_heatmaps(axes, heatmaps, title_prefix='Heatmaps'):
    """Visualize heatmaps in a grid."""
    n_joints = heatmaps.shape[0]

    for i in range(min(n_joints, len(axes))):
        axes[i].imshow(heatmaps[i], cmap='hot')
        axes[i].set_title(f'{i}: {MO2CAP2_JOINTS[i]}', fontsize=8)
        axes[i].axis('off')


def visualize_training_sample(sample, save_path=None):
    """Visualize a training sample with all annotations."""
    fig = plt.figure(figsize=(20, 14))

    # Title
    fig.suptitle(f"Mo2Cap2 Training - Chunk {sample['chunk_id']:04d}, Sample {sample['sample_id']}",
                 fontsize=14, fontweight='bold')

    # Use GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 0.8, 0.8])

    # Row 1: Images with 2D keypoints and 3D poses
    ax1 = fig.add_subplot(gs[0, 0])
    visualize_2d(ax1, sample['image'], sample['annot_2d'], 'Original Image + 2D')

    ax2 = fig.add_subplot(gs[0, 1])
    visualize_2d(ax2, sample['zoom_image'], None, 'Zoomed Image')

    # 3D pose (two views)
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    visualize_3d(ax3, sample['annot_3d'], '3D Pose (Front)', elev=0, azim=0)

    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    visualize_3d(ax4, sample['annot_3d'], '3D Pose (Side)', elev=0, azim=90)

    # Row 2-3: Heatmaps (8 per row, 15 total)
    gs_hm = GridSpec(2, 8, figure=fig, top=0.45, bottom=0.02, hspace=0.3)
    n_joints = sample['heatmaps'].shape[0]

    for i in range(n_joints):
        row = i // 8
        col = i % 8
        ax = fig.add_subplot(gs_hm[row, col])
        ax.imshow(sample['heatmaps'][i], cmap='hot')
        ax.set_title(f'{i}: {MO2CAP2_JOINTS[i]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_test_sample(sample, save_path=None):
    """Visualize a test sample."""
    fig = plt.figure(figsize=(16, 6))

    # Title
    fig.suptitle(f"Mo2Cap2 Test - {sample['test_set']}, {sample['img_name']}",
                 fontsize=14, fontweight='bold')

    # Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(sample['image'])
    ax1.set_title('Test Image')
    ax1.axis('off')

    # 3D pose (two views)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    visualize_3d(ax2, sample['annot_3d'], '3D GT (Front)', elev=0, azim=0)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    visualize_3d(ax3, sample['annot_3d'], '3D GT (Side)', elev=0, azim=90)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_annotation_stats(data_root='/mnt/sdb2/mo2cap2_dataset'):
    """Visualize annotation statistics from the cache."""
    cache_path = os.path.join(data_root, 'training_data', 'annotations_cache.h5')

    with h5py.File(cache_path, 'r') as f:
        keypoint3d = f['keypoint3d'][:]  # (530000, 15, 3)
        hmd_info = f['hmd_info'][:]      # (530000, 9)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mo2Cap2 Annotation Statistics (530k samples)', fontsize=14, fontweight='bold')

    # 3D keypoint distribution per axis
    for i, (ax, axis_name) in enumerate(zip(axes[0], ['X', 'Y', 'Z'])):
        data = keypoint3d[:, :, i].flatten()
        ax.hist(data, bins=100, alpha=0.7, color='steelblue')
        ax.set_title(f'3D Keypoint {axis_name} Distribution')
        ax.set_xlabel(f'{axis_name} (meters)')
        ax.set_ylabel('Count')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        ax.legend()

    # HMD info distributions
    hmd_labels = ['RH_x', 'RH_y', 'RH_z', 'LH_x', 'LH_y', 'LH_z', 'HandDist', 'RDist', 'LDist']
    for i, (ax, label) in enumerate(zip(axes[1], hmd_labels[:3])):
        data = hmd_info[:, i]
        ax.hist(data, bins=100, alpha=0.7, color='coral')
        ax.set_title(f'HMD Info: {label}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('mo2cap2_stats.png', dpi=150, bbox_inches='tight')
    print("Saved: mo2cap2_stats.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Mo2Cap2 dataset')
    parser.add_argument('--data-root', type=str, default='/mnt/sdb2/mo2cap2_dataset',
                        help='Path to Mo2Cap2 dataset')
    parser.add_argument('--chunk', type=int, default=None,
                        help='Training chunk ID (1-530)')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample ID within chunk (0-999) or test set')
    parser.add_argument('--test', type=str, default=None,
                        choices=['olek_outdoor', 'weipeng_studio'],
                        help='Test set name')
    parser.add_argument('--save', action='store_true',
                        help='Save visualization instead of showing')
    parser.add_argument('--stats', action='store_true',
                        help='Show annotation statistics')
    parser.add_argument('--output-dir', type=str, default='output_mo2cap2_vis',
                        help='Output directory for saved images')

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.stats:
        visualize_annotation_stats(args.data_root)
        return

    if args.test:
        # Visualize test sample
        sample = load_test_sample(args.test, args.sample, args.data_root)
        save_path = os.path.join(args.output_dir, f'test_{args.test}_{args.sample:04d}.png') if args.save else None
        visualize_test_sample(sample, save_path)
    elif args.chunk:
        # Visualize training sample
        sample = load_training_sample(args.chunk, args.sample, args.data_root)
        save_path = os.path.join(args.output_dir, f'train_chunk{args.chunk:04d}_{args.sample:04d}.png') if args.save else None
        visualize_training_sample(sample, save_path)
    else:
        # Default: show first sample from first chunk
        print("No chunk or test set specified. Showing first training sample...")
        sample = load_training_sample(1, 0, args.data_root)
        visualize_training_sample(sample)


if __name__ == '__main__':
    main()
