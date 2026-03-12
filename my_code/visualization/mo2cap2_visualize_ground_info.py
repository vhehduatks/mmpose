#!/usr/bin/env python3
"""
Visualize Mo2Cap2 3D Pose with Ground Reference Information

This script visualizes how the ground reference info should be computed
in the Mo2Cap2 dataset using a BODY-RELATIVE direction.

Since the 3D joints are in camera coordinates (not world coordinates),
none of the X/Y/Z axes directly correspond to the user's height.
Instead, we use the body's own orientation:

**Vector a** (body downward direction):
    - From Neck[0] toward pelvis center
    - pelvis_center = average(L.UpLeg[11], R.UpLeg[7])
    - a = normalize(pelvis_center - neck)

**Ground Reference Point**:
    - The farthest toe from Neck in the direction of vector a
    - Compare: L.ToeBase[14] and R.ToeBase[10]
    - Use whichever has larger projection onto vector a

**ground_head_info**:
    - Distance from Neck to ground reference point along vector a

**ground_ctrl_info** (for hands):
    - Distance from Hand to ground reference point along vector a

Usage:
    python my_code/visualization/mo2cap2_visualize_ground_info.py --num-samples 5
    python my_code/visualization/mo2cap2_visualize_ground_info.py --sample-idx 100 200 300
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mo2Cap2 constants
MO2CAP2_NUM_JOINTS = 15
MO2CAP2_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # Right arm: Neck -> RightArm -> RightForeArm -> RightHand
    (0, 4), (4, 5), (5, 6),      # Left arm: Neck -> LeftArm -> LeftForeArm -> LeftHand
    (0, 7), (7, 8), (8, 9), (9, 10),    # Right leg: Neck -> RightUpLeg -> RightLeg -> RightFoot -> RightToeBase
    (0, 11), (11, 12), (12, 13), (13, 14),  # Left leg: Neck -> LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToeBase
]

MO2CAP2_JOINT_NAMES = [
    'Neck', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'
]

# Joint indices
NECK_IDX = 0
RIGHT_ARM_IDX = 1
LEFT_ARM_IDX = 4
RIGHT_HAND_IDX = 3
LEFT_HAND_IDX = 6
RIGHT_UPLEG_IDX = 7
LEFT_UPLEG_IDX = 11
RIGHT_FOOT_IDX = 9
LEFT_FOOT_IDX = 13
RIGHT_TOE_IDX = 10
LEFT_TOE_IDX = 14


def load_sample_from_h5(data_root, sample_idx):
    """Load a single sample from Mo2Cap2 H5 chunks."""
    chunk_files = sorted([
        f for f in os.listdir(data_root)
        if f.startswith('mo2cap2_chunk_') and f.endswith('.hdf5')
    ])

    if len(chunk_files) == 0:
        raise ValueError(f'No chunk files found in {data_root}')

    cumsum = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(data_root, chunk_file)
        with h5py.File(chunk_path, 'r') as hf:
            chunk_size = hf['Images'].shape[0]
            if cumsum + chunk_size > sample_idx:
                local_idx = sample_idx - cumsum
                image = hf['Images'][local_idx]
                keypoint2d = hf['Annot2D'][local_idx]
                keypoint3d = hf['Annot3D'][local_idx]
                image = np.transpose(image, (1, 2, 0))
                return {
                    'image': image,
                    'keypoint2d': keypoint2d,
                    'keypoint3d': keypoint3d,
                    'chunk_file': chunk_file,
                    'local_idx': local_idx,
                    'global_idx': sample_idx
                }
            cumsum += chunk_size

    raise ValueError(f'Sample index {sample_idx} out of range (total: {cumsum})')


def make_root_relative(keypoint3d):
    """Convert to root-relative coordinates (Neck at origin)."""
    neck = keypoint3d[NECK_IDX].copy()
    return keypoint3d - neck


def compute_ground_info_body_relative(keypoint3d):
    """Compute ground reference using body-relative direction.

    The key insight: since data is in camera coordinates, we use the
    body's own orientation to determine the "downward" direction.

    Vector a (body downward direction):
        a = normalize(pelvis_center - neck)
        where pelvis_center = average(L.UpLeg, R.UpLeg)

    Ground reference point:
        The toe (L.ToeBase or R.ToeBase) that is farthest from Neck
        in the direction of vector a.

    Args:
        keypoint3d: (15, 3) 3D keypoints in camera coordinates

    Returns:
        dict with ground info
    """
    # Make root-relative (Neck at origin)
    kp = make_root_relative(keypoint3d)

    neck = kp[NECK_IDX]  # [0, 0, 0] after root-relative
    left_upleg = kp[LEFT_UPLEG_IDX]
    right_upleg = kp[RIGHT_UPLEG_IDX]
    left_hand = kp[LEFT_HAND_IDX]
    right_hand = kp[RIGHT_HAND_IDX]
    left_toe = kp[LEFT_TOE_IDX]
    right_toe = kp[RIGHT_TOE_IDX]

    # Compute pelvis center (average of UpLegs)
    pelvis_center = (left_upleg + right_upleg) / 2

    # Vector a: direction from Neck to pelvis (body "downward" direction)
    vec_a = pelvis_center - neck  # neck is [0,0,0], so this is just pelvis_center
    vec_a_norm = np.linalg.norm(vec_a)
    if vec_a_norm < 1e-8:
        # Fallback if pelvis is at neck (shouldn't happen)
        vec_a_unit = np.array([0, 1, 0])
    else:
        vec_a_unit = vec_a / vec_a_norm

    # Project toes onto vector a to find which is "lower" (farther along a)
    # Projection: proj_a(p) = (p · a_unit) where p is relative to neck
    left_toe_proj = np.dot(left_toe - neck, vec_a_unit)
    right_toe_proj = np.dot(right_toe - neck, vec_a_unit)

    # Ground reference is the farthest toe along vector a
    if left_toe_proj >= right_toe_proj:
        ground_ref_point = left_toe
        ground_ref_proj = left_toe_proj
        ground_ref_name = 'L.ToeBase'
    else:
        ground_ref_point = right_toe
        ground_ref_proj = right_toe_proj
        ground_ref_name = 'R.ToeBase'

    # ground_head_info: distance from Neck to ground ref along vector a
    # Since neck is at origin, this is just the projection value
    ground_head_info = ground_ref_proj

    # ground_ctrl_info: distance from Hand to ground ref along vector a
    # = ground_ref_proj - hand_proj_onto_a
    left_hand_proj = np.dot(left_hand - neck, vec_a_unit)
    right_hand_proj = np.dot(right_hand - neck, vec_a_unit)

    left_hand_ground_dist = ground_ref_proj - left_hand_proj
    right_hand_ground_dist = ground_ref_proj - right_hand_proj

    return {
        'keypoint3d_processed': kp,
        'vec_a': vec_a,
        'vec_a_unit': vec_a_unit,
        'pelvis_center': pelvis_center,
        'ground_ref_point': ground_ref_point,
        'ground_ref_proj': ground_ref_proj,
        'ground_ref_name': ground_ref_name,
        'left_toe_proj': left_toe_proj,
        'right_toe_proj': right_toe_proj,
        'ground_head_info': ground_head_info,
        'left_hand_proj': left_hand_proj,
        'right_hand_proj': right_hand_proj,
        'left_hand_ground_dist': left_hand_ground_dist,
        'right_hand_ground_dist': right_hand_ground_dist,
        'neck_pos': neck,
        'left_hand_pos': left_hand,
        'right_hand_pos': right_hand,
        'left_toe_pos': left_toe,
        'right_toe_pos': right_toe,
    }


def visualize_ground_info(sample, output_path=None, show=True):
    """Visualize 3D pose with body-relative ground reference."""
    keypoint3d_raw = sample['keypoint3d']
    image = sample['image']

    # Compute ground info with body-relative method
    ground_info = compute_ground_info_body_relative(keypoint3d_raw)

    kp = ground_info['keypoint3d_processed']
    vec_a_unit = ground_info['vec_a_unit']
    ground_ref_proj = ground_info['ground_ref_proj']

    # Create figure
    fig = plt.figure(figsize=(18, 7))

    # 1. Image with 2D keypoints
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)

    keypoint2d = sample['keypoint2d']
    for i, (x, y) in enumerate(keypoint2d):
        color = 'lime' if i < 7 else 'red'
        ax1.scatter(x, y, c=color, s=30, zorder=5)

    for (i, j) in MO2CAP2_SKELETON:
        color = 'lime' if i < 7 and j < 7 else 'red'
        ax1.plot([keypoint2d[i, 0], keypoint2d[j, 0]],
                 [keypoint2d[i, 1], keypoint2d[j, 1]], c=color, linewidth=1.5)

    ax1.set_title(f'2D View (Sample {sample["global_idx"]})')
    ax1.axis('off')

    # 2. 3D pose - camera view (X vs Z)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    _draw_3d_pose_with_body_axis(ax2, kp, ground_info,
                                  title='3D Pose with Body-Relative Ground')

    # 3. 3D pose - side view showing vector a
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    _draw_3d_pose_with_body_axis(ax3, kp, ground_info,
                                  title='Side View (showing body axis)',
                                  view_elev=0, view_azim=0)

    # Add info text box
    info_text = (
        f"Body-Relative Ground Reference:\n"
        f"─────────────────────────────────\n"
        f"Vector a (Neck→Pelvis direction):\n"
        f"  [{vec_a_unit[0]:.3f}, {vec_a_unit[1]:.3f}, {vec_a_unit[2]:.3f}]\n"
        f"\n"
        f"Ground reference: {ground_info['ground_ref_name']}\n"
        f"  L.ToeBase proj: {ground_info['left_toe_proj']:.1f}mm\n"
        f"  R.ToeBase proj: {ground_info['right_toe_proj']:.1f}mm\n"
        f"\n"
        f"Ground Info (Enhanced HMD):\n"
        f"  ground_head_info: {ground_info['ground_head_info']:.1f}mm\n"
        f"  L.hand→ground: {ground_info['left_hand_ground_dist']:.1f}mm\n"
        f"  R.hand→ground: {ground_info['right_hand_ground_dist']:.1f}mm"
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    if show:
        plt.show()
    else:
        plt.close()


def _draw_3d_pose_with_body_axis(ax, kp, ground_info, title='',
                                  view_elev=20, view_azim=-60):
    """Draw 3D pose with body axis vector and ground plane."""
    vec_a_unit = ground_info['vec_a_unit']
    ground_ref_proj = ground_info['ground_ref_proj']
    pelvis_center = ground_info['pelvis_center']

    # Draw skeleton
    for (i, j) in MO2CAP2_SKELETON:
        color = 'blue' if i < 7 and j < 7 else 'darkred'
        ax.plot([kp[i, 0], kp[j, 0]],
                [kp[i, 1], kp[j, 1]],
                [kp[i, 2], kp[j, 2]],
                c=color, linewidth=2)

    # Draw joints
    colors = ['blue' if i < 7 else 'darkred' for i in range(15)]
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c=colors, s=50, marker='o')

    # Mark special joints
    ax.scatter(*kp[NECK_IDX], c='cyan', s=120, marker='*',
               label='Neck (Origin)', zorder=10)
    ax.scatter(*pelvis_center, c='magenta', s=80, marker='D',
               label='Pelvis Center', zorder=10)
    ax.scatter(*kp[LEFT_HAND_IDX], c='lime', s=80, marker='s',
               label='L.Hand', zorder=10)
    ax.scatter(*kp[RIGHT_HAND_IDX], c='orange', s=80, marker='s',
               label='R.Hand', zorder=10)
    ax.scatter(*kp[LEFT_TOE_IDX], c='green', s=60, marker='^',
               label='L.ToeBase', zorder=10)
    ax.scatter(*kp[RIGHT_TOE_IDX], c='red', s=60, marker='^',
               label='R.ToeBase', zorder=10)

    neck = kp[NECK_IDX]
    left_hand = kp[LEFT_HAND_IDX]
    right_hand = kp[RIGHT_HAND_IDX]

    # Ground center point (neck projected to ground plane along vector a)
    ground_center = neck + vec_a_unit * ground_ref_proj

    # Create ground plane (rectangle perpendicular to vector a)
    # Find two orthogonal vectors perpendicular to vec_a_unit
    if abs(vec_a_unit[0]) < 0.9:
        perp1 = np.cross(vec_a_unit, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(vec_a_unit, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(vec_a_unit, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # Ground plane size
    plane_size = 400  # mm

    # Four corners of the ground plane
    corner1 = ground_center + plane_size * perp1 + plane_size * perp2
    corner2 = ground_center + plane_size * perp1 - plane_size * perp2
    corner3 = ground_center - plane_size * perp1 - plane_size * perp2
    corner4 = ground_center - plane_size * perp1 + plane_size * perp2

    ground_verts = [
        [corner1[0], corner1[1], corner1[2]],
        [corner2[0], corner2[1], corner2[2]],
        [corner3[0], corner3[1], corner3[2]],
        [corner4[0], corner4[1], corner4[2]],
    ]
    ground_plane = Poly3DCollection([ground_verts], alpha=0.3, facecolor='brown',
                                     edgecolor='saddlebrown', linewidth=2)
    ax.add_collection3d(ground_plane)

    # Draw Neck→Ground (body axis) as dashed line (same style as hand-ground)
    ax.plot([neck[0], ground_center[0]],
            [neck[1], ground_center[1]],
            [neck[2], ground_center[2]],
            c='cyan', linewidth=2, linestyle='--',
            label=f'Neck→Ground ({ground_info["ground_head_info"]:.0f}mm)')

    # Point on ground plane directly "below" each hand (along vector a)
    left_hand_ground_point = left_hand + ground_info['left_hand_ground_dist'] * vec_a_unit
    right_hand_ground_point = right_hand + ground_info['right_hand_ground_dist'] * vec_a_unit

    # Draw L.Hand→Ground (parallel to vector a)
    ax.plot([left_hand[0], left_hand_ground_point[0]],
            [left_hand[1], left_hand_ground_point[1]],
            [left_hand[2], left_hand_ground_point[2]],
            c='lime', linewidth=2, linestyle='--',
            label=f'L.Hand→Ground ({ground_info["left_hand_ground_dist"]:.0f}mm)')

    # Draw R.Hand→Ground (parallel to vector a)
    ax.plot([right_hand[0], right_hand_ground_point[0]],
            [right_hand[1], right_hand_ground_point[1]],
            [right_hand[2], right_hand_ground_point[2]],
            c='orange', linewidth=2, linestyle='--',
            label=f'R.Hand→Ground ({ground_info["right_hand_ground_dist"]:.0f}mm)')

    # Mark the ground plane intersection points
    ax.scatter(*ground_center, c='cyan', s=60, marker='x', zorder=10)
    ax.scatter(*left_hand_ground_point, c='lime', s=40, marker='x', zorder=10)
    ax.scatter(*right_hand_ground_point, c='orange', s=40, marker='x', zorder=10)

    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)

    ax.view_init(elev=view_elev, azim=view_azim)
    ax.legend(loc='upper left', fontsize=6)


def main():
    parser = argparse.ArgumentParser(description='Visualize Mo2Cap2 Body-Relative Ground Info')
    parser.add_argument('--data-root', type=str,
                        default='/mnt/sdb2/mo2cap2_dataset/training_data',
                        help='Path to Mo2Cap2 H5 chunk directory')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of random samples to visualize')
    parser.add_argument('--sample-idx', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--output-dir', type=str, default='output_ground_info_vis',
                        help='Output directory for saved figures')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures (only save)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample_idx is not None:
        sample_indices = args.sample_idx
    else:
        np.random.seed(42)
        sample_indices = np.random.randint(1000, 500000, size=args.num_samples).tolist()

    print(f'Visualizing {len(sample_indices)} samples from {args.data_root}')
    print(f'Sample indices: {sample_indices}')
    print()
    print('Body-Relative Ground Reference Method:')
    print('  1. Vector a = direction from Neck[0] to Pelvis center')
    print('     Pelvis center = average(L.UpLeg[11], R.UpLeg[7])')
    print('  2. Ground ref = farthest toe along vector a')
    print('     Compare L.ToeBase[14] vs R.ToeBase[10]')
    print('  3. ground_head_info = Neck-to-ground distance along a')
    print('  4. ground_ctrl_info = Hand-to-ground distance along a')
    print()

    for idx in sample_indices:
        try:
            sample = load_sample_from_h5(args.data_root, idx)
            output_path = os.path.join(args.output_dir, f'ground_info_sample_{idx:06d}.png')
            visualize_ground_info(sample, output_path=output_path, show=not args.no_show)

            ground_info = compute_ground_info_body_relative(sample['keypoint3d'])

            print(f'\nSample {idx} ({sample["chunk_file"]}):')
            print(f'  Vector a (body axis): [{ground_info["vec_a_unit"][0]:.3f}, '
                  f'{ground_info["vec_a_unit"][1]:.3f}, {ground_info["vec_a_unit"][2]:.3f}]')
            print(f'  Ground reference: {ground_info["ground_ref_name"]}')
            print(f'    L.ToeBase projection: {ground_info["left_toe_proj"]:.1f}mm')
            print(f'    R.ToeBase projection: {ground_info["right_toe_proj"]:.1f}mm')
            print(f'  Ground Info:')
            print(f'    ground_head_info (Neck→Ground): {ground_info["ground_head_info"]:.1f}mm')
            print(f'    L.Hand→Ground: {ground_info["left_hand_ground_dist"]:.1f}mm')
            print(f'    R.Hand→Ground: {ground_info["right_hand_ground_dist"]:.1f}mm')

        except Exception as e:
            print(f'Error processing sample {idx}: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
