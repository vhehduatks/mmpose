"""
xR-EgoPose Ground Info Visualization V2

Enhanced visualization that supports:
1. Both GT and prediction visualization
2. Integration with mmpose val pipeline via hook
3. Standalone usage for debugging

Usage (standalone):
    python my_code/visualization/visualize_egopose_ground_info_v2.py --sample 0
    python my_code/visualization/visualize_egopose_ground_info_v2.py --sample 0 --save

Usage (in val pipeline):
    # Add to config:
    custom_hooks = [
        dict(type='GroundInfoVisualizationHook', interval=100, mode='both_from_ground')
    ]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py
from typing import Optional, Dict, Any, Tuple

# xR-EgoPose skeleton definition (16 joints)
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

# Joint indices
SPINE2_IDX = 0
HEAD_IDX = 1
LEFT_HAND_IDX = 4
RIGHT_HAND_IDX = 7
LEFT_UPLEG_IDX = 8
RIGHT_UPLEG_IDX = 12
LEFT_FOOT_IDX = 10
RIGHT_FOOT_IDX = 14
LEFT_TOE_IDX = 11
RIGHT_TOE_IDX = 15

# Colors for GT and Pred
COLORS = {
    'left': '#3498db',           # Blue
    'right': '#e74c3c',          # Red
    'center': '#2ecc71',         # Green
    'ground': '#95a5a6',         # Gray
    'head_line': '#9b59b6',      # Purple
    'left_hand_line': '#1abc9c', # Teal
    'right_hand_line': '#e67e22', # Orange
    # Prediction colors (lighter/different)
    'pred_left': '#85c1e9',      # Light Blue
    'pred_right': '#f1948a',     # Light Red
    'pred_center': '#82e0aa',    # Light Green
    'pred_skeleton': '#aab7b8', # Light Gray
}


def get_bone_color(parent: int, child: int, is_pred: bool = False) -> str:
    """Get color based on bone side and GT/Pred status."""
    left_joints = {2, 3, 4, 8, 9, 10, 11}
    right_joints = {5, 6, 7, 12, 13, 14, 15}

    if is_pred:
        if parent in left_joints or child in left_joints:
            return COLORS['pred_left']
        elif parent in right_joints or child in right_joints:
            return COLORS['pred_right']
        else:
            return COLORS['pred_center']
    else:
        if parent in left_joints or child in left_joints:
            return COLORS['left']
        elif parent in right_joints or child in right_joints:
            return COLORS['right']
        else:
            return COLORS['center']


def load_egopose_sample(sample_id: int, cache_path: str) -> Dict[str, Any]:
    """Load a sample from xR-EgoPose H5 cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    with h5py.File(cache_path, 'r') as f:
        keypoint3d = f['keypoint3d'][sample_id][0]  # (16, 3)
        keypoints = f['keypoints'][sample_id][0]    # (16, 2)
        hmd_info = f['hmd_info'][sample_id][0]      # (9,)

        if 'images' in f:
            image = f['images'][sample_id]  # (256, 256, 3)
        else:
            image = None

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


def compute_ground_info(keypoint3d: np.ndarray, mode: str = 'both_from_ground') -> Dict[str, Any]:
    """Compute ground info from 3D keypoints using body-relative direction."""
    spine2 = keypoint3d[SPINE2_IDX]
    head = keypoint3d[HEAD_IDX]
    left_hand = keypoint3d[LEFT_HAND_IDX]
    right_hand = keypoint3d[RIGHT_HAND_IDX]
    left_upleg = keypoint3d[LEFT_UPLEG_IDX]
    right_upleg = keypoint3d[RIGHT_UPLEG_IDX]
    left_toe = keypoint3d[LEFT_TOE_IDX]
    right_toe = keypoint3d[RIGHT_TOE_IDX]

    # Pelvis center
    pelvis_center = (left_upleg + right_upleg) / 2

    # Vector A = body axis (Spine2 -> Pelvis)
    vec_a = pelvis_center - spine2
    vec_a_norm = np.linalg.norm(vec_a)
    if vec_a_norm > 1e-6:
        vec_a_unit = vec_a / vec_a_norm
    else:
        vec_a_unit = np.array([0, 0, 1])

    # Project toes to find ground
    left_toe_proj = np.dot(left_toe - spine2, vec_a_unit)
    right_toe_proj = np.dot(right_toe - spine2, vec_a_unit)
    ground_ref_proj = max(left_toe_proj, right_toe_proj)

    # Heights from ground
    spine2_proj = 0
    head_proj = np.dot(head - spine2, vec_a_unit)
    left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
    right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

    spine2_from_ground = ground_ref_proj - spine2_proj
    head_from_ground = ground_ref_proj - head_proj
    left_hand_from_ground = ground_ref_proj - left_hand_proj
    right_hand_from_ground = ground_ref_proj - right_hand_proj

    head_torso_dist = np.linalg.norm(spine2 - pelvis_center)

    return {
        'vec_a_unit': vec_a_unit,
        'pelvis_center': pelvis_center,
        'ground_ref_proj': ground_ref_proj,
        'spine2_from_ground': spine2_from_ground,
        'head_from_ground': head_from_ground,
        'left_hand_from_ground': left_hand_from_ground,
        'right_hand_from_ground': right_hand_from_ground,
        'head_torso_dist': head_torso_dist,
        'spine2': spine2,
        'head': head,
        'left_hand': left_hand,
        'right_hand': right_hand,
        'left_toe': left_toe,
        'right_toe': right_toe,
    }


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute MPJPE (Mean Per Joint Position Error)."""
    per_joint_error = np.linalg.norm(gt - pred, axis=1)  # (16,)
    mpjpe = np.mean(per_joint_error)
    return mpjpe, per_joint_error


def visualize_gt_and_pred(
    sample: Dict[str, Any],
    pred_keypoint3d: Optional[np.ndarray] = None,
    mode: str = 'both_from_ground',
    save_path: Optional[str] = None,
    show_error: bool = True
) -> None:
    """Visualize both GT and prediction 3D poses with ground info.

    Args:
        sample: dict from load_egopose_sample (contains GT)
        pred_keypoint3d: (16, 3) predicted 3D keypoints (optional)
        mode: Ground info mode
        save_path: If provided, save figure
        show_error: Whether to show per-joint error
    """
    gt_keypoint3d = sample['keypoint3d']
    gt_ground_info = compute_ground_info(gt_keypoint3d, mode)

    # Compute prediction ground info if available
    pred_ground_info = None
    mpjpe = None
    per_joint_error = None
    if pred_keypoint3d is not None:
        pred_ground_info = compute_ground_info(pred_keypoint3d, mode)
        mpjpe, per_joint_error = compute_mpjpe(gt_keypoint3d, pred_keypoint3d)

    # Figure layout: 2 rows x 3 cols
    # Row 1: 2D image, GT 3D view 1, GT 3D view 2
    # Row 2: Error info, Pred 3D view 1, Pred 3D view 2 (or comparison)
    if pred_keypoint3d is not None:
        fig = plt.figure(figsize=(18, 14))
        title_suffix = f" | MPJPE: {mpjpe*1000:.2f}mm" if mpjpe else ""
    else:
        fig = plt.figure(figsize=(18, 10))
        title_suffix = ""

    fig.suptitle(
        f"xR-EgoPose Ground Info - Sample {sample['sample_id']} ({sample['action']}){title_suffix}",
        fontsize=14, fontweight='bold'
    )

    # ===== Row 1: GT =====
    # 2D Image
    ax1 = fig.add_subplot(2 if pred_keypoint3d is not None else 1, 3, 1)
    _draw_2d_image(ax1, sample, per_joint_error if show_error else None)

    # GT 3D View 1
    ax2 = fig.add_subplot(2 if pred_keypoint3d is not None else 1, 3, 2, projection='3d')
    _draw_3d_pose(ax2, gt_keypoint3d, gt_ground_info, mode, elev=20, azim=45,
                  title='GT 3D Pose (View 1)', is_pred=False)

    # GT 3D View 2
    ax3 = fig.add_subplot(2 if pred_keypoint3d is not None else 1, 3, 3, projection='3d')
    _draw_3d_pose(ax3, gt_keypoint3d, gt_ground_info, mode, elev=10, azim=135,
                  title='GT 3D Pose (View 2)', is_pred=False)

    # ===== Row 2: Pred (if available) =====
    if pred_keypoint3d is not None:
        # Error info panel
        ax4 = fig.add_subplot(2, 3, 4)
        _draw_error_info(ax4, gt_ground_info, pred_ground_info, per_joint_error, mode)

        # Pred 3D View 1
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        _draw_3d_pose(ax5, pred_keypoint3d, pred_ground_info, mode, elev=20, azim=45,
                      title='Pred 3D Pose (View 1)', is_pred=True,
                      gt_keypoint3d=gt_keypoint3d)

        # Pred 3D View 2
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        _draw_3d_pose(ax6, pred_keypoint3d, pred_ground_info, mode, elev=10, azim=135,
                      title='Pred 3D Pose (View 2)', is_pred=True,
                      gt_keypoint3d=gt_keypoint3d)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def _draw_2d_image(ax, sample: Dict, per_joint_error: Optional[np.ndarray] = None) -> None:
    """Draw 2D image with keypoints."""
    if sample['image'] is not None:
        img_h, img_w = sample['image'].shape[:2]
        ax.imshow(sample['image'])

        keypoints = sample['keypoints']

        # Draw skeleton
        for parent, child in EGOPOSE_SKELETON:
            px, py = keypoints[parent]
            cx, cy = keypoints[child]
            if (0 <= px < img_w and 0 <= py < img_h and
                0 <= cx < img_w and 0 <= cy < img_h):
                color = get_bone_color(parent, child)
                ax.plot([px, cx], [py, cy], color=color, linewidth=2, alpha=0.8)

        # Draw joints
        for i, (x, y) in enumerate(keypoints):
            if not (0 <= x < img_w and 0 <= y < img_h):
                continue

            if i in {2, 3, 4, 8, 9, 10, 11}:
                color = COLORS['left']
            elif i in {5, 6, 7, 12, 13, 14, 15}:
                color = COLORS['right']
            else:
                color = COLORS['center']

            ax.scatter(x, y, c=color, s=40, zorder=5, edgecolors='white', linewidths=1)

            # Label with error if available
            if per_joint_error is not None:
                label_text = f'{i}:{per_joint_error[i]*1000:.0f}'
            else:
                label_text = f'{i}'

            if x < img_w - 35:
                label_x, ha = x + 6, 'left'
            else:
                label_x, ha = x - 6, 'right'
            label_y = y - 6 if y > 12 else y + 12

            ax.text(label_x, label_y, label_text, fontsize=7, color='white',
                    fontweight='bold', zorder=6, ha=ha,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.9, edgecolor='none'))

        ax.set_title('2D Image + Keypoints' + (' (error in mm)' if per_joint_error is not None else ''))
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
    else:
        ax.text(0.5, 0.5, 'No image available', ha='center', va='center', fontsize=12)
        ax.set_title('2D Image')
    ax.axis('off')


def _draw_3d_pose(
    ax,
    keypoint3d: np.ndarray,
    ground_info: Dict,
    mode: str,
    elev: float = 20,
    azim: float = 45,
    title: str = '3D Pose',
    is_pred: bool = False,
    gt_keypoint3d: Optional[np.ndarray] = None
) -> None:
    """Draw 3D pose with ground plane."""
    vec_a_unit = ground_info['vec_a_unit']
    spine2 = ground_info['spine2']
    ground_ref_proj = ground_info['ground_ref_proj']

    # Draw GT skeleton (faded) if this is pred view
    if is_pred and gt_keypoint3d is not None:
        for parent, child in EGOPOSE_SKELETON:
            ax.plot3D(
                [gt_keypoint3d[parent, 0], gt_keypoint3d[child, 0]],
                [gt_keypoint3d[parent, 1], gt_keypoint3d[child, 1]],
                [gt_keypoint3d[parent, 2], gt_keypoint3d[child, 2]],
                color='gray', linewidth=1, alpha=0.3
            )

    # Draw skeleton
    for parent, child in EGOPOSE_SKELETON:
        color = get_bone_color(parent, child, is_pred)
        ax.plot3D(
            [keypoint3d[parent, 0], keypoint3d[child, 0]],
            [keypoint3d[parent, 1], keypoint3d[child, 1]],
            [keypoint3d[parent, 2], keypoint3d[child, 2]],
            color=color, linewidth=2
        )

    # Draw joints
    for i, (x, y, z) in enumerate(keypoint3d):
        if i in {2, 3, 4, 8, 9, 10, 11}:
            color = COLORS['pred_left'] if is_pred else COLORS['left']
        elif i in {5, 6, 7, 12, 13, 14, 15}:
            color = COLORS['pred_right'] if is_pred else COLORS['right']
        else:
            color = COLORS['pred_center'] if is_pred else COLORS['center']
        ax.scatter3D(x, y, z, c=color, s=40, edgecolors='white', linewidths=0.5)

    # Ground plane
    ground_point = spine2 + vec_a_unit * ground_ref_proj

    if abs(vec_a_unit[0]) < 0.9:
        perp1 = np.cross(vec_a_unit, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(vec_a_unit, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(vec_a_unit, perp1)

    plane_size = 0.5
    corners = []
    for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        corner = ground_point + s1 * plane_size * perp1 + s2 * plane_size * perp2
        corners.append(corner)

    verts = [corners]
    ground_plane = Poly3DCollection(verts, alpha=0.3, facecolor=COLORS['ground'], edgecolor='gray')
    ax.add_collection3d(ground_plane)

    ax.scatter3D(ground_point[0], ground_point[1], ground_point[2],
                 c=COLORS['ground'], s=80, marker='^', edgecolors='black', linewidths=1)

    # Height lines (only for GT view, pred view shows comparison)
    if not is_pred:
        # Root to ground
        spine2_ground = spine2 + vec_a_unit * ground_ref_proj
        ax.plot3D(
            [spine2[0], spine2_ground[0]],
            [spine2[1], spine2_ground[1]],
            [spine2[2], spine2_ground[2]],
            color=COLORS['head_line'], linewidth=3, linestyle='--', alpha=0.8
        )
        mid_spine2 = (spine2 + spine2_ground) / 2
        ax.text(mid_spine2[0] + 0.08, mid_spine2[1], mid_spine2[2],
                f'Root: {ground_info["spine2_from_ground"]:.2f}m',
                fontsize=8, color=COLORS['head_line'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Hand lines
        if mode in ['hand_from_ground', 'both_from_ground']:
            left_hand = ground_info['left_hand']
            right_hand = ground_info['right_hand']

            left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
            right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

            left_hand_to_ground = ground_ref_proj - left_hand_proj
            right_hand_to_ground = ground_ref_proj - right_hand_proj

            left_hand_ground_point = left_hand + vec_a_unit * left_hand_to_ground
            right_hand_ground_point = right_hand + vec_a_unit * right_hand_to_ground

            ax.plot3D(
                [left_hand[0], left_hand_ground_point[0]],
                [left_hand[1], left_hand_ground_point[1]],
                [left_hand[2], left_hand_ground_point[2]],
                color=COLORS['left_hand_line'], linewidth=3, linestyle='--', alpha=0.8
            )
            ax.scatter3D(left_hand_ground_point[0], left_hand_ground_point[1], left_hand_ground_point[2],
                         c=COLORS['left_hand_line'], s=50, marker='s', edgecolors='black', linewidths=1)

            ax.plot3D(
                [right_hand[0], right_hand_ground_point[0]],
                [right_hand[1], right_hand_ground_point[1]],
                [right_hand[2], right_hand_ground_point[2]],
                color=COLORS['right_hand_line'], linewidth=3, linestyle='--', alpha=0.8
            )
            ax.scatter3D(right_hand_ground_point[0], right_hand_ground_point[1], right_hand_ground_point[2],
                         c=COLORS['right_hand_line'], s=50, marker='s', edgecolors='black', linewidths=1)

    # Axis settings
    all_points = np.vstack([keypoint3d, [ground_point]])
    if gt_keypoint3d is not None:
        all_points = np.vstack([all_points, gt_keypoint3d])

    ax.set_xlim([all_points[:, 0].min() - 0.3, all_points[:, 0].max() + 0.3])
    ax.set_ylim([all_points[:, 1].min() - 0.3, all_points[:, 1].max() + 0.3])
    ax.set_zlim([all_points[:, 2].min() - 0.3, all_points[:, 2].max() + 0.3])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)


def _draw_error_info(
    ax,
    gt_ground_info: Dict,
    pred_ground_info: Dict,
    per_joint_error: np.ndarray,
    mode: str
) -> None:
    """Draw error information panel."""
    ax.axis('off')

    # Per-joint error table
    text_lines = [
        f"{'='*50}",
        f"Per-Joint Errors (mm)",
        f"{'='*50}",
    ]

    for i, joint_name in enumerate(EGOPOSE_JOINTS):
        error_mm = per_joint_error[i] * 1000
        bar = '|' + '#' * min(int(error_mm / 5), 20)
        text_lines.append(f"{i:2d} {joint_name:15s}: {error_mm:6.1f}mm {bar}")

    text_lines.append(f"{'='*50}")
    text_lines.append(f"MPJPE: {np.mean(per_joint_error)*1000:.2f}mm")
    text_lines.append(f"{'='*50}")

    # Ground info comparison
    text_lines.append(f"\nGround Info Comparison (GT vs Pred)")
    text_lines.append(f"{'='*50}")
    text_lines.append(f"Root from ground:  GT={gt_ground_info['spine2_from_ground']:.3f}m  "
                     f"Pred={pred_ground_info['spine2_from_ground']:.3f}m")

    if mode in ['head_from_ground', 'both_from_ground']:
        text_lines.append(f"Head from ground:  GT={gt_ground_info['head_from_ground']:.3f}m  "
                         f"Pred={pred_ground_info['head_from_ground']:.3f}m")

    if mode in ['hand_from_ground', 'both_from_ground']:
        text_lines.append(f"L.Hand from ground: GT={gt_ground_info['left_hand_from_ground']:.3f}m  "
                         f"Pred={pred_ground_info['left_hand_from_ground']:.3f}m")
        text_lines.append(f"R.Hand from ground: GT={gt_ground_info['right_hand_from_ground']:.3f}m  "
                         f"Pred={pred_ground_info['right_hand_from_ground']:.3f}m")

    text = '\n'.join(text_lines)
    ax.text(0.05, 0.95, text, fontsize=8, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_title('Error Analysis')


def visualize_from_pipeline_data(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    keypoint3d_gt: np.ndarray,
    keypoint3d_pred: np.ndarray,
    action: str = 'unknown',
    sample_id: int = 0,
    mode: str = 'both_from_ground',
    save_path: Optional[str] = None
) -> None:
    """Visualize from pipeline data (for use in hooks).

    Args:
        image: (H, W, 3) RGB image
        keypoints_2d: (16, 2) 2D keypoints
        keypoint3d_gt: (16, 3) GT 3D keypoints
        keypoint3d_pred: (16, 3) predicted 3D keypoints
        action: action name
        sample_id: sample index
        mode: ground info mode
        save_path: save path
    """
    sample = {
        'keypoint3d': keypoint3d_gt,
        'keypoints': keypoints_2d,
        'hmd_info': np.zeros(9),  # Not used for visualization
        'image': image,
        'action': action,
        'sample_id': sample_id,
    }
    visualize_gt_and_pred(sample, keypoint3d_pred, mode=mode, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(description='Visualize xR-EgoPose ground info (V2)')
    parser.add_argument('--cache', type=str,
                        default='/mnt/dataset_vol/h5cache/train_cache_v2.h5',
                        help='Path to H5 cache file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--mode', type=str, default='both_from_ground',
                        choices=['head_from_ground', 'hand_from_ground', 'both_from_ground', 'ground_reference'],
                        help='Ground info mode')
    parser.add_argument('--save', action='store_true',
                        help='Save visualization')
    parser.add_argument('--output-dir', type=str, default='output_egopose_ground_vis_v2',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples to visualize')
    parser.add_argument('--simulate-pred', action='store_true',
                        help='Simulate prediction with noise for testing')
    parser.add_argument('--noise-std', type=float, default=0.03,
                        help='Noise standard deviation for simulated prediction (meters)')

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        sample_id = args.sample + i
        try:
            sample = load_egopose_sample(sample_id, args.cache)

            # Simulate prediction if requested
            pred_keypoint3d = None
            if args.simulate_pred:
                pred_keypoint3d = sample['keypoint3d'] + np.random.randn(16, 3) * args.noise_std

            save_path = os.path.join(args.output_dir, f'ground_info_v2_{sample_id:06d}.png') if args.save else None
            visualize_gt_and_pred(sample, pred_keypoint3d, mode=args.mode, save_path=save_path)
        except Exception as e:
            print(f"Error visualizing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
