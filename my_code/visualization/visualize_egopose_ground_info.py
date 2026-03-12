"""
xR-EgoPose Ground Info Visualization

Visualize 3D poses with ground reference information to verify the enhanced HMD data.
Uses body-relative direction (Vector A) for correct ground reference computation.

The key insight is that in camera coordinates, NO axis corresponds to "up/down".
Instead, we use the body's own skeletal structure to define "downward":
- Vector A: from Spine2 (root) toward Pelvis center (body axis)
- Ground: farthest toe projection along Vector A
- Heights: projection distances along Vector A

Usage:
    python my_code/visualization/visualize_egopose_ground_info.py --sample 0
    python my_code/visualization/visualize_egopose_ground_info.py --sample 0 --save
    python my_code/visualization/visualize_egopose_ground_info.py --sample 0 --mode both_from_ground
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py

# xR-EgoPose skeleton definition (16 joints)
# Based on mmpose/datasets/datasets/body3d/egopose_info.py
EGOPOSE_JOINTS = [
    'Spine2',       # 0 (root)
    'Head',         # 1
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

# Skeleton connections (parent, child)
EGOPOSE_SKELETON = [
    (0, 1),    # Spine2 -> Head
    (0, 2),    # Spine2 -> LeftArm
    (2, 3),    # LeftArm -> LeftForeArm
    (3, 4),    # LeftForeArm -> LeftHand
    (0, 5),    # Spine2 -> RightArm
    (5, 6),    # RightArm -> RightForeArm
    (6, 7),    # RightForeArm -> RightHand
    (0, 8),    # Spine2 -> LeftUpLeg
    (8, 9),    # LeftUpLeg -> LeftLeg
    (9, 10),   # LeftLeg -> LeftFoot
    (10, 11),  # LeftFoot -> LeftToeBase
    (0, 12),   # Spine2 -> RightUpLeg
    (12, 13),  # RightUpLeg -> RightLeg
    (13, 14),  # RightLeg -> RightFoot
    (14, 15),  # RightFoot -> RightToeBase
]

# Joint indices for ground info computation (xR-EgoPose)
SPINE2_IDX = 0       # Root
HEAD_IDX = 1
LEFT_HAND_IDX = 4
RIGHT_HAND_IDX = 7
LEFT_UPLEG_IDX = 8   # For pelvis center
RIGHT_UPLEG_IDX = 12 # For pelvis center
LEFT_FOOT_IDX = 10
RIGHT_FOOT_IDX = 14
LEFT_TOE_IDX = 11    # For ground reference
RIGHT_TOE_IDX = 15   # For ground reference

# Colors
COLORS = {
    'left': '#3498db',           # Blue
    'right': '#e74c3c',          # Red
    'center': '#2ecc71',         # Green
    'ground': '#95a5a6',         # Gray
    'head_line': '#9b59b6',      # Purple
    'left_hand_line': '#1abc9c', # Teal (distinct from skeleton blue)
    'right_hand_line': '#e67e22', # Orange (distinct from skeleton red)
}


def get_bone_color(parent, child):
    """Get color based on bone side."""
    left_joints = {2, 3, 4, 8, 9, 10, 11}
    right_joints = {5, 6, 7, 12, 13, 14, 15}

    if parent in left_joints or child in left_joints:
        return COLORS['left']
    elif parent in right_joints or child in right_joints:
        return COLORS['right']
    else:
        return COLORS['center']


def load_egopose_sample(sample_id, cache_path='/mnt/dataset_vol/h5cache/train_cache_with_images.h5'):
    """Load a sample from xR-EgoPose H5 cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    with h5py.File(cache_path, 'r') as f:
        # keypoint3d shape: (N, 1, 16, 3)
        keypoint3d = f['keypoint3d'][sample_id][0]  # (16, 3)
        keypoints = f['keypoints'][sample_id][0]    # (16, 2)
        hmd_info = f['hmd_info'][sample_id][0]      # (9,)

        # Load image if available
        if 'images' in f:
            image = f['images'][sample_id]  # (256, 256, 3)
        else:
            image = None

        # Load action if available
        if 'actions' in f:
            action = f['actions'][sample_id]
            if isinstance(action, bytes):
                action = action.decode('utf-8')
        else:
            action = 'unknown'

    return {
        'keypoint3d': keypoint3d,
        'keypoints': keypoints,
        'hmd_info': hmd_info,
        'image': image,
        'action': action,
        'sample_id': sample_id,
    }


def compute_ground_info(keypoint3d, mode='both_from_ground'):
    """Compute ground info from 3D keypoints using body-relative direction.

    CORRECT METHOD: Uses Vector A (body axis from Spine2 to Pelvis) to define
    the "downward" direction, instead of assuming any coordinate axis is height.

    Args:
        keypoint3d: (16, 3) array of 3D keypoints (camera coordinates)
        mode: 'head_from_ground', 'hand_from_ground', 'both_from_ground', 'ground_reference'

    Returns:
        dict with ground info values computed along body axis
    """
    # Extract key joints
    spine2 = keypoint3d[SPINE2_IDX]      # Root
    head = keypoint3d[HEAD_IDX]
    left_hand = keypoint3d[LEFT_HAND_IDX]
    right_hand = keypoint3d[RIGHT_HAND_IDX]
    left_upleg = keypoint3d[LEFT_UPLEG_IDX]
    right_upleg = keypoint3d[RIGHT_UPLEG_IDX]
    left_toe = keypoint3d[LEFT_TOE_IDX]
    right_toe = keypoint3d[RIGHT_TOE_IDX]

    # ========================================
    # CORRECT: Body-relative direction (Vector A)
    # ========================================

    # Step 1: Compute pelvis center (midpoint of hip joints)
    pelvis_center = (left_upleg + right_upleg) / 2

    # Step 2: Vector A = direction from Spine2 toward Pelvis (body "downward")
    vec_a = pelvis_center - spine2
    vec_a_norm = np.linalg.norm(vec_a)
    if vec_a_norm > 1e-6:
        vec_a_unit = vec_a / vec_a_norm
    else:
        vec_a_unit = np.array([0, 0, 1])  # Fallback

    # Step 3: Project toes onto Vector A to find ground reference
    left_toe_proj = np.dot(left_toe - spine2, vec_a_unit)
    right_toe_proj = np.dot(right_toe - spine2, vec_a_unit)

    # Ground = farthest toe along body axis
    ground_ref_proj = max(left_toe_proj, right_toe_proj)

    # Step 4: Compute heights from ground (along body axis)
    # Spine2 projection is 0 (it's the origin of our vector)
    spine2_proj = 0
    head_proj = np.dot(head - spine2, vec_a_unit)
    left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
    right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

    # Heights from ground (ground_ref_proj - joint_proj)
    spine2_from_ground = ground_ref_proj - spine2_proj  # = ground_ref_proj
    head_from_ground = ground_ref_proj - head_proj
    left_hand_from_ground = ground_ref_proj - left_hand_proj
    right_hand_from_ground = ground_ref_proj - right_hand_proj

    # Head-torso distance (Euclidean, for ground_reference mode)
    head_torso_dist = np.linalg.norm(spine2 - pelvis_center)

    # ========================================
    # Also compute OLD (incorrect) Y-axis method for comparison
    # ========================================
    old_ground_y = min(keypoint3d[LEFT_FOOT_IDX, 1], keypoint3d[RIGHT_FOOT_IDX, 1])
    old_spine2_from_ground = -old_ground_y

    return {
        # Correct (body-axis) values
        'vec_a_unit': vec_a_unit,
        'pelvis_center': pelvis_center,
        'ground_ref_proj': ground_ref_proj,
        'spine2_from_ground': spine2_from_ground,
        'head_from_ground': head_from_ground,
        'left_hand_from_ground': left_hand_from_ground,
        'right_hand_from_ground': right_hand_from_ground,
        'head_torso_dist': head_torso_dist,
        # Joint positions
        'spine2': spine2,
        'head': head,
        'left_hand': left_hand,
        'right_hand': right_hand,
        'left_toe': left_toe,
        'right_toe': right_toe,
        # Old (incorrect) values for comparison
        'old_ground_y': old_ground_y,
        'old_spine2_from_ground': old_spine2_from_ground,
    }


def visualize_ground_info(sample, mode='both_from_ground', save_path=None):
    """Visualize 3D pose with ground info using body-relative direction.

    Args:
        sample: dict from load_egopose_sample
        mode: Ground info mode to visualize
        save_path: If provided, save figure instead of showing
    """
    keypoint3d = sample['keypoint3d']
    ground_info = compute_ground_info(keypoint3d, mode)

    fig = plt.figure(figsize=(18, 10))

    # Title
    fig.suptitle(f"xR-EgoPose Ground Info (Body-Axis Method) - Sample {sample['sample_id']} ({sample['action']})",
                 fontsize=14, fontweight='bold')

    # Left subplot: 2D image with keypoints (if available)
    ax1 = fig.add_subplot(1, 3, 1)
    if sample['image'] is not None:
        img_h, img_w = sample['image'].shape[:2]
        ax1.imshow(sample['image'])

        # Draw 2D keypoints with skeleton connections
        keypoints = sample['keypoints']

        # Draw skeleton connections first (behind joints)
        for parent, child in EGOPOSE_SKELETON:
            px, py = keypoints[parent]
            cx, cy = keypoints[child]
            # Only draw if both points are within image bounds
            if (0 <= px < img_w and 0 <= py < img_h and
                0 <= cx < img_w and 0 <= cy < img_h):
                color = get_bone_color(parent, child)
                ax1.plot([px, cx], [py, cy], color=color, linewidth=2, alpha=0.8)

        # Draw joints and labels (only for points within image bounds)
        for i, (x, y) in enumerate(keypoints):
            # Skip if outside image bounds
            if not (0 <= x < img_w and 0 <= y < img_h):
                continue

            if i in {2, 3, 4, 8, 9, 10, 11}:
                color = COLORS['left']
            elif i in {5, 6, 7, 12, 13, 14, 15}:
                color = COLORS['right']
            else:
                color = COLORS['center']
            ax1.scatter(x, y, c=color, s=40, zorder=5, edgecolors='white', linewidths=1)

            # Add joint labels with position adjustment to stay inside image
            # Position label to the right if there's space, otherwise to the left
            if x < img_w - 25:
                label_x = x + 6
                ha = 'left'
            else:
                label_x = x - 6
                ha = 'right'
            # Position label above if there's space, otherwise below
            if y > 12:
                label_y = y - 6
            else:
                label_y = y + 12
            ax1.text(label_x, label_y, f'{i}', fontsize=7, color='white',
                     fontweight='bold', zorder=6, ha=ha,
                     bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.9, edgecolor='none'))

        ax1.set_title('2D Image + Keypoints')
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)  # Invert Y axis for image coordinates
    else:
        ax1.text(0.5, 0.5, 'No image available', ha='center', va='center', fontsize=12)
        ax1.set_title('2D Image')
    ax1.axis('off')

    # Middle subplot: 3D pose (front view)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_3d_pose_with_ground(ax2, keypoint3d, ground_info, mode, elev=20, azim=45, title='3D Pose (View 1)')

    # Right subplot: 3D pose (side view)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_3d_pose_with_ground(ax3, keypoint3d, ground_info, mode, elev=10, azim=135, title='3D Pose (View 2)')

    # Add text annotation with ground info values
    info_text = create_info_text(ground_info, mode)
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def draw_3d_pose_with_ground(ax, keypoint3d, ground_info, mode, elev=15, azim=0, title='3D Pose'):
    """Draw 3D pose with body-axis ground plane and height lines."""
    vec_a_unit = ground_info['vec_a_unit']
    spine2 = ground_info['spine2']
    ground_ref_proj = ground_info['ground_ref_proj']

    # Draw skeleton
    for parent, child in EGOPOSE_SKELETON:
        color = get_bone_color(parent, child)
        ax.plot3D(
            [keypoint3d[parent, 0], keypoint3d[child, 0]],
            [keypoint3d[parent, 1], keypoint3d[child, 1]],
            [keypoint3d[parent, 2], keypoint3d[child, 2]],
            color=color, linewidth=2
        )

    # Draw joints
    for i, (x, y, z) in enumerate(keypoint3d):
        if i in {2, 3, 4, 8, 9, 10, 11}:
            color = COLORS['left']
        elif i in {5, 6, 7, 12, 13, 14, 15}:
            color = COLORS['right']
        else:
            color = COLORS['center']
        ax.scatter3D(x, y, z, c=color, s=40, edgecolors='white', linewidths=0.5)

    # Ground point = spine2 + vec_a_unit * ground_ref_proj
    ground_point = spine2 + vec_a_unit * ground_ref_proj

    # Create orthonormal basis for the ground plane
    # Find two vectors perpendicular to vec_a_unit
    if abs(vec_a_unit[0]) < 0.9:
        perp1 = np.cross(vec_a_unit, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(vec_a_unit, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(vec_a_unit, perp1)

    # Draw ground plane as a square
    plane_size = 0.5
    corners = []
    for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        corner = ground_point + s1 * plane_size * perp1 + s2 * plane_size * perp2
        corners.append(corner)

    # Plot ground plane
    verts = [corners]
    ground_plane = Poly3DCollection(verts, alpha=0.3, facecolor=COLORS['ground'], edgecolor='gray')
    ax.add_collection3d(ground_plane)

    # Mark ground point with label
    ax.scatter3D(ground_point[0], ground_point[1], ground_point[2],
                 c=COLORS['ground'], s=80, marker='^', edgecolors='black', linewidths=1)
    ax.text(ground_point[0], ground_point[1], ground_point[2] - 0.08,
            'Ground Plane', fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Draw Spine2-to-ground line with label
    spine2_ground = spine2 + vec_a_unit * ground_ref_proj
    ax.plot3D(
        [spine2[0], spine2_ground[0]],
        [spine2[1], spine2_ground[1]],
        [spine2[2], spine2_ground[2]],
        color=COLORS['head_line'], linewidth=3, linestyle='--', alpha=0.8
    )
    # Add label at midpoint
    mid_spine2 = (spine2 + spine2_ground) / 2
    ax.text(mid_spine2[0] + 0.08, mid_spine2[1], mid_spine2[2],
            f'Root: {ground_info["spine2_from_ground"]:.2f}m',
            fontsize=8, color=COLORS['head_line'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Hand lines (for hand_from_ground and both_from_ground modes)
    if mode in ['hand_from_ground', 'both_from_ground']:
        left_hand = ground_info['left_hand']
        right_hand = ground_info['right_hand']

        # Compute hand projections along Vector A (from spine2)
        left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
        right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

        # Hand-to-ground distance along Vector A
        left_hand_to_ground = ground_ref_proj - left_hand_proj
        right_hand_to_ground = ground_ref_proj - right_hand_proj

        # Ground intersection points: hand + vec_a_unit * distance_to_ground
        # This draws a line PARALLEL to Vector A from the hand to the ground plane
        left_hand_ground_point = left_hand + vec_a_unit * left_hand_to_ground
        right_hand_ground_point = right_hand + vec_a_unit * right_hand_to_ground

        # Left hand -> ground (TEAL color for left)
        ax.plot3D(
            [left_hand[0], left_hand_ground_point[0]],
            [left_hand[1], left_hand_ground_point[1]],
            [left_hand[2], left_hand_ground_point[2]],
            color=COLORS['left_hand_line'], linewidth=3, linestyle='--', alpha=0.8
        )
        ax.scatter3D(left_hand_ground_point[0], left_hand_ground_point[1], left_hand_ground_point[2],
                     c=COLORS['left_hand_line'], s=50, marker='s', edgecolors='black', linewidths=1)
        # Label for left hand
        mid_left = (left_hand + left_hand_ground_point) / 2
        ax.text(mid_left[0] - 0.12, mid_left[1], mid_left[2],
                f'L.Hand: {ground_info["left_hand_from_ground"]:.2f}m',
                fontsize=8, color=COLORS['left_hand_line'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Right hand -> ground (ORANGE color for right)
        ax.plot3D(
            [right_hand[0], right_hand_ground_point[0]],
            [right_hand[1], right_hand_ground_point[1]],
            [right_hand[2], right_hand_ground_point[2]],
            color=COLORS['right_hand_line'], linewidth=3, linestyle='--', alpha=0.8
        )
        ax.scatter3D(right_hand_ground_point[0], right_hand_ground_point[1], right_hand_ground_point[2],
                     c=COLORS['right_hand_line'], s=50, marker='s', edgecolors='black', linewidths=1)
        # Label for right hand
        mid_right = (right_hand + right_hand_ground_point) / 2
        ax.text(mid_right[0] + 0.08, mid_right[1], mid_right[2],
                f'R.Hand: {ground_info["right_hand_from_ground"]:.2f}m',
                fontsize=8, color=COLORS['right_hand_line'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Set axis properties
    all_points = np.vstack([keypoint3d, [ground_point]])

    ax.set_xlim([all_points[:, 0].min() - 0.3, all_points[:, 0].max() + 0.3])
    ax.set_ylim([all_points[:, 1].min() - 0.3, all_points[:, 1].max() + 0.3])
    ax.set_zlim([all_points[:, 2].min() - 0.3, all_points[:, 2].max() + 0.3])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)


def create_info_text(ground_info, mode):
    """Create text annotation with ground info values."""
    text = f"Ground Info (mode: {mode})\n"
    text += "=" * 50 + "\n"
    text += "Body-Axis Method (Correct)\n"
    text += "-" * 50 + "\n"
    text += f"Ground ref (toe projection): {ground_info['ground_ref_proj']:.3f} m\n"
    text += "-" * 50 + "\n"
    text += f"[Purple] Root from ground:   {ground_info['spine2_from_ground']:.3f} m\n"

    if mode in ['hand_from_ground', 'both_from_ground']:
        text += f"[Teal]   L.Hand from ground: {ground_info['left_hand_from_ground']:.3f} m\n"
        text += f"[Orange] R.Hand from ground: {ground_info['right_hand_from_ground']:.3f} m\n"

    if mode == 'ground_reference':
        text += "-" * 50 + "\n"
        text += f"Head-Torso distance:         {ground_info['head_torso_dist']:.3f} m\n"

    return text


def main():
    parser = argparse.ArgumentParser(description='Visualize xR-EgoPose ground info')
    parser.add_argument('--cache', type=str,
                        default='/mnt/dataset_vol/h5cache/train_cache_with_images.h5',
                        help='Path to H5 cache file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--mode', type=str, default='both_from_ground',
                        choices=['head_from_ground', 'hand_from_ground', 'both_from_ground', 'ground_reference'],
                        help='Ground info mode to visualize')
    parser.add_argument('--save', action='store_true',
                        help='Save visualization instead of showing')
    parser.add_argument('--output-dir', type=str, default='output_egopose_ground_vis',
                        help='Output directory for saved images')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples to visualize (starting from --sample)')

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        sample_id = args.sample + i
        try:
            sample = load_egopose_sample(sample_id, args.cache)
            save_path = os.path.join(args.output_dir, f'ground_info_{sample_id:06d}.png') if args.save else None
            visualize_ground_info(sample, mode=args.mode, save_path=save_path)
        except Exception as e:
            print(f"Error visualizing sample {sample_id}: {e}")


if __name__ == '__main__':
    main()
