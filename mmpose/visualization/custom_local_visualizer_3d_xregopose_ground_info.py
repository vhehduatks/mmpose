# Copyright (c) OpenMMLab. All rights reserved.
"""3D Pose Visualizer with Ground Info for XR EgoPose dataset.

Visualizes GT and Pred 3D poses with body-axis ground reference information.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mmengine.dist import master_only
from mmengine.structures import InstanceData

from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from . import PoseLocalVisualizer


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

EGOPOSE_JOINTS = [
    'Spine2', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg',
    'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
]

# Joint indices
SPINE2_IDX = 0
HEAD_IDX = 1
LEFT_HAND_IDX = 4
RIGHT_HAND_IDX = 7
LEFT_UPLEG_IDX = 8
RIGHT_UPLEG_IDX = 12
LEFT_TOE_IDX = 11
RIGHT_TOE_IDX = 15

# Colors
COLORS = {
    'gt_left': '#3498db',        # Blue
    'gt_right': '#e74c3c',       # Red
    'gt_center': '#2ecc71',      # Green
    'pred_left': '#85c1e9',      # Light Blue
    'pred_right': '#f1948a',     # Light Red
    'pred_center': '#82e0aa',    # Light Green
    'ground': '#95a5a6',         # Gray
    'head_line': '#9b59b6',      # Purple
    'left_hand_line': '#1abc9c', # Teal
    'right_hand_line': '#e67e22', # Orange
}

# Keypoint colors (RGB normalized)
EGOPOSE_KPT_COLORS = np.array([
    [51, 153, 255],   # Spine2
    [51, 153, 255],   # Head
    [52, 152, 219],   # LeftArm
    [52, 152, 219],   # LeftForeArm
    [52, 152, 219],   # LeftHand
    [231, 76, 60],    # RightArm
    [231, 76, 60],    # RightForeArm
    [231, 76, 60],    # RightHand
    [52, 152, 219],   # LeftUpLeg
    [52, 152, 219],   # LeftLeg
    [52, 152, 219],   # LeftFoot
    [52, 152, 219],   # LeftToeBase
    [231, 76, 60],    # RightUpLeg
    [231, 76, 60],    # RightLeg
    [231, 76, 60],    # RightFoot
    [231, 76, 60],    # RightToeBase
]) / 255.0

EGOPOSE_LINK_COLORS = np.array([
    [51, 153, 255],   # Spine2 -> Head
    [52, 152, 219],   # Spine2 -> LeftArm
    [52, 152, 219],   # LeftArm -> LeftForeArm
    [52, 152, 219],   # LeftForeArm -> LeftHand
    [231, 76, 60],    # Spine2 -> RightArm
    [231, 76, 60],    # RightArm -> RightForeArm
    [231, 76, 60],    # RightForeArm -> RightHand
    [52, 152, 219],   # Spine2 -> LeftUpLeg
    [52, 152, 219],   # LeftUpLeg -> LeftLeg
    [52, 152, 219],   # LeftLeg -> LeftFoot
    [52, 152, 219],   # LeftFoot -> LeftToeBase
    [231, 76, 60],    # Spine2 -> RightUpLeg
    [231, 76, 60],    # RightUpLeg -> RightLeg
    [231, 76, 60],    # RightLeg -> RightFoot
    [231, 76, 60],    # RightFoot -> RightToeBase
]) / 255.0


def compute_ground_info(keypoint3d: np.ndarray) -> Dict:
    """Compute ground info using body-axis method."""
    spine2 = keypoint3d[SPINE2_IDX]
    head = keypoint3d[HEAD_IDX]
    left_hand = keypoint3d[LEFT_HAND_IDX]
    right_hand = keypoint3d[RIGHT_HAND_IDX]
    left_upleg = keypoint3d[LEFT_UPLEG_IDX]
    right_upleg = keypoint3d[RIGHT_UPLEG_IDX]
    left_toe = keypoint3d[LEFT_TOE_IDX]
    right_toe = keypoint3d[RIGHT_TOE_IDX]

    pelvis_center = (left_upleg + right_upleg) / 2
    vec_a = pelvis_center - spine2
    vec_a_norm = np.linalg.norm(vec_a)
    if vec_a_norm > 1e-6:
        vec_a_unit = vec_a / vec_a_norm
    else:
        vec_a_unit = np.array([0, 0, 1])

    left_toe_proj = np.dot(left_toe - spine2, vec_a_unit)
    right_toe_proj = np.dot(right_toe - spine2, vec_a_unit)
    ground_ref_proj = max(left_toe_proj, right_toe_proj)

    head_proj = np.dot(head - spine2, vec_a_unit)
    left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
    right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

    return {
        'vec_a_unit': vec_a_unit,
        'spine2': spine2,
        'ground_ref_proj': ground_ref_proj,
        'spine2_from_ground': ground_ref_proj,
        'head_from_ground': ground_ref_proj - head_proj,
        'left_hand_from_ground': ground_ref_proj - left_hand_proj,
        'right_hand_from_ground': ground_ref_proj - right_hand_proj,
        'left_hand': left_hand,
        'right_hand': right_hand,
    }


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute MPJPE."""
    per_joint_error = np.linalg.norm(gt - pred, axis=1)
    mpjpe = np.mean(per_joint_error)
    return mpjpe, per_joint_error


@VISUALIZERS.register_module()
class CustomPose3dLocalVisualizer_xregopose_ground_info(PoseLocalVisualizer):
    """3D Pose Visualizer with Ground Info for XR EgoPose.

    Visualizes:
    - Row 1: 2D Image + Keypoints, GT 3D with ground info
    - Row 2: Error panel, Pred 3D with ground info (GT overlay)

    Args:
        name (str): Name of visualizer instance
        ground_info_mode (str): Ground info mode ('head_from_ground', 'hand_from_ground', 'both_from_ground')
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: int = 4,
                 radius: int = 3,
                 alpha: float = 0.8,
                 ground_info_mode: str = 'both_from_ground',
                 **kwargs):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            line_width=line_width,
            radius=radius,
            alpha=alpha,
            **kwargs
        )
        self.ground_info_mode = ground_info_mode
        self._egopose_kpt_colors = EGOPOSE_KPT_COLORS
        self._egopose_link_colors = EGOPOSE_LINK_COLORS

    def _draw_2d_image(self, ax, image: np.ndarray, keypoints_2d: np.ndarray,
                       title: str = '2D Image + Keypoints',
                       show_joint_idx: bool = True) -> None:
        """Draw 2D image with keypoints.

        Args:
            ax: Matplotlib axis
            image: Input image
            keypoints_2d: 2D keypoints array
            title: Title for the subplot
            show_joint_idx: Whether to show joint index labels
        """
        ax.imshow(image)

        if keypoints_2d is None:
            ax.set_title(f'{title} (N/A)', fontsize=10)
            ax.axis('off')
            return

        if keypoints_2d.ndim == 3:
            keypoints_2d = keypoints_2d[0]

        h, w = image.shape[:2]

        # Draw skeleton
        for idx, (i, j) in enumerate(EGOPOSE_SKELETON):
            if i < len(keypoints_2d) and j < len(keypoints_2d):
                pt1, pt2 = keypoints_2d[i], keypoints_2d[j]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                            color=self._egopose_link_colors[idx], linewidth=2)

        # Draw joints
        for i, kpt in enumerate(keypoints_2d):
            if 0 <= kpt[0] < w and 0 <= kpt[1] < h:
                ax.scatter(kpt[0], kpt[1], c=[self._egopose_kpt_colors[i]],
                          s=50, marker='o', edgecolors='white', linewidths=1, zorder=5)

                # Show joint index label if enabled
                if show_joint_idx:
                    label = f'{i}'

                    if kpt[0] < w - 30:
                        label_x, ha = kpt[0] + 5, 'left'
                    else:
                        label_x, ha = kpt[0] - 5, 'right'
                    label_y = kpt[1] - 5 if kpt[1] > 12 else kpt[1] + 12

                    ax.text(label_x, label_y, label, fontsize=6, color='white',
                            fontweight='bold', ha=ha,
                            bbox=dict(boxstyle='round,pad=0.1',
                                      facecolor=self._egopose_kpt_colors[i], alpha=0.9, edgecolor='none'))

        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def _draw_3d_pose_with_ground(self, ax, keypoint3d: np.ndarray, ground_info: Dict,
                                   title: str = '3D Pose', is_pred: bool = False,
                                   gt_keypoint3d: Optional[np.ndarray] = None,
                                   elev: float = 20, azim: float = 45) -> None:
        """Draw 3D pose with ground plane."""
        if keypoint3d is None:
            ax.set_title(title)
            return

        if keypoint3d.ndim == 3:
            keypoint3d = keypoint3d[0]

        vec_a_unit = ground_info['vec_a_unit']
        spine2 = ground_info['spine2']
        ground_ref_proj = ground_info['ground_ref_proj']

        # Draw GT skeleton faded if this is pred view
        if is_pred and gt_keypoint3d is not None:
            if gt_keypoint3d.ndim == 3:
                gt_keypoint3d = gt_keypoint3d[0]
            for i, j in EGOPOSE_SKELETON:
                ax.plot3D(
                    [gt_keypoint3d[i, 0], gt_keypoint3d[j, 0]],
                    [gt_keypoint3d[i, 1], gt_keypoint3d[j, 1]],
                    [gt_keypoint3d[i, 2], gt_keypoint3d[j, 2]],
                    color='gray', linewidth=1, alpha=0.4, linestyle='--'
                )

        # Skeleton color adjustment for pred
        alpha_val = 0.7 if is_pred else 1.0
        kpt_colors = self._egopose_kpt_colors * (0.7 if is_pred else 1.0)
        link_colors = self._egopose_link_colors * (0.7 if is_pred else 1.0)

        # Draw skeleton
        for idx, (i, j) in enumerate(EGOPOSE_SKELETON):
            ax.plot3D(
                [keypoint3d[i, 0], keypoint3d[j, 0]],
                [keypoint3d[i, 1], keypoint3d[j, 1]],
                [keypoint3d[i, 2], keypoint3d[j, 2]],
                color=link_colors[idx], linewidth=2, alpha=alpha_val
            )

        # Draw joints
        ax.scatter3D(keypoint3d[:, 0], keypoint3d[:, 1], keypoint3d[:, 2],
                     c=kpt_colors, s=40, marker='o', edgecolors='white', linewidths=0.5)

        # Ground plane
        ground_point = spine2 + vec_a_unit * ground_ref_proj

        if abs(vec_a_unit[0]) < 0.9:
            perp1 = np.cross(vec_a_unit, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(vec_a_unit, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(vec_a_unit, perp1)

        plane_size = 0.4
        corners = []
        for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            corner = ground_point + s1 * plane_size * perp1 + s2 * plane_size * perp2
            corners.append(corner)

        verts = [corners]
        ground_plane = Poly3DCollection(verts, alpha=0.25, facecolor=COLORS['ground'], edgecolor='gray')
        ax.add_collection3d(ground_plane)

        ax.scatter3D(ground_point[0], ground_point[1], ground_point[2],
                     c=COLORS['ground'], s=60, marker='^', edgecolors='black', linewidths=1)

        # Height lines (only for GT view)
        if not is_pred:
            # Root to ground
            spine2_ground = spine2 + vec_a_unit * ground_ref_proj
            ax.plot3D(
                [spine2[0], spine2_ground[0]],
                [spine2[1], spine2_ground[1]],
                [spine2[2], spine2_ground[2]],
                color=COLORS['head_line'], linewidth=2, linestyle='--', alpha=0.8
            )

            # Hand lines
            if self.ground_info_mode in ['hand_from_ground', 'both_from_ground']:
                left_hand = ground_info['left_hand']
                right_hand = ground_info['right_hand']

                left_hand_proj = np.dot(left_hand - spine2, vec_a_unit)
                right_hand_proj = np.dot(right_hand - spine2, vec_a_unit)

                left_hand_ground = left_hand + vec_a_unit * (ground_ref_proj - left_hand_proj)
                right_hand_ground = right_hand + vec_a_unit * (ground_ref_proj - right_hand_proj)

                ax.plot3D(
                    [left_hand[0], left_hand_ground[0]],
                    [left_hand[1], left_hand_ground[1]],
                    [left_hand[2], left_hand_ground[2]],
                    color=COLORS['left_hand_line'], linewidth=2, linestyle='--', alpha=0.8
                )

                ax.plot3D(
                    [right_hand[0], right_hand_ground[0]],
                    [right_hand[1], right_hand_ground[1]],
                    [right_hand[2], right_hand_ground[2]],
                    color=COLORS['right_hand_line'], linewidth=2, linestyle='--', alpha=0.8
                )

        # Axis settings
        all_points = keypoint3d.copy()
        if gt_keypoint3d is not None:
            gt_kpts = gt_keypoint3d[0] if gt_keypoint3d.ndim == 3 else gt_keypoint3d
            all_points = np.vstack([all_points, gt_kpts])

        center = all_points.mean(axis=0)
        max_range = np.abs(all_points - center).max() * 1.3

        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=elev, azim=azim)

    def _create_visualization(self,
                              image: np.ndarray,
                              pred_kpts_2d: Optional[np.ndarray] = None,
                              gt_kpts_2d: Optional[np.ndarray] = None,
                              pred_kpts_3d: Optional[np.ndarray] = None,
                              gt_kpts_3d: Optional[np.ndarray] = None) -> np.ndarray:
        """Create full visualization with ground info.

        Layout:
        - Row 1, Col 1: 2D Image + Keypoints GT
        - Row 1, Col 2: GT 3D Pose + Ground
        - Row 2, Col 1: 2D Image + Keypoints Pred
        - Row 2, Col 2: Pred 3D Pose + Ground (with GT overlay)
        """
        plt.ioff()

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

        # Compute ground info and MPJPE
        gt_ground_info = None
        pred_ground_info = None
        mpjpe = None

        if gt_kpts_3d is not None:
            gt_3d = gt_kpts_3d[0] if gt_kpts_3d.ndim == 3 else gt_kpts_3d
            gt_ground_info = compute_ground_info(gt_3d)

        if pred_kpts_3d is not None:
            pred_3d = pred_kpts_3d[0] if pred_kpts_3d.ndim == 3 else pred_kpts_3d
            pred_ground_info = compute_ground_info(pred_3d)

            if gt_kpts_3d is not None:
                mpjpe, _ = compute_mpjpe(gt_3d, pred_3d)

        # Use same view angle for both 3D plots
        view_elev = 20
        view_azim = 45

        # Row 1, Col 1: 2D Image + Keypoints GT
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_2d_image(ax1, image, gt_kpts_2d, title='2D Image + Keypoints GT')

        # Row 1, Col 2: GT 3D with ground info
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        if gt_ground_info is not None:
            self._draw_3d_pose_with_ground(ax2, gt_kpts_3d, gt_ground_info,
                                            title='GT 3D Pose + Ground', is_pred=False,
                                            elev=view_elev, azim=view_azim)
        else:
            ax2.set_title('GT 3D (N/A)')

        # Row 2, Col 1: 2D Image + Keypoints Pred
        ax3 = fig.add_subplot(gs[1, 0])
        self._draw_2d_image(ax3, image, pred_kpts_2d, title='2D Image + Keypoints Pred')

        # Row 2, Col 2: Pred 3D with ground info (GT overlay)
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        if pred_ground_info is not None:
            self._draw_3d_pose_with_ground(ax4, pred_kpts_3d, pred_ground_info,
                                            title=f'Pred 3D (MPJPE: {mpjpe*1000:.1f}mm)' if mpjpe else 'Pred 3D',
                                            is_pred=True, gt_keypoint3d=gt_kpts_3d,
                                            elev=view_elev, azim=view_azim)
        else:
            ax4.set_title('Pred 3D (N/A)')

        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.get_size_inches() * fig.get_dpi()
        img = img.reshape(int(h), int(w), 3)
        plt.close(fig)

        return img

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_2d: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0,
                       **kwargs) -> np.ndarray:
        """Draw datasample with ground info visualization."""
        pred_kpts_2d = None
        gt_kpts_2d = None
        pred_kpts_3d = None
        gt_kpts_3d = None

        # Extract prediction data
        if draw_pred and 'pred_instances' in data_sample:
            pred = data_sample.pred_instances

            if 'keypoints' in pred:
                kpts = pred.get('transformed_keypoints', pred.keypoints)
                if hasattr(kpts, 'cpu'):
                    kpts = kpts.cpu().numpy()
                pred_kpts_2d = kpts

            if 'keypoint_3d' in pred:
                kpts_3d = pred.keypoint_3d
                if hasattr(kpts_3d, 'cpu'):
                    kpts_3d = kpts_3d.cpu().numpy()
                pred_kpts_3d = kpts_3d

        # Extract ground truth data
        if draw_gt and 'gt_instances' in data_sample:
            gt = data_sample.gt_instances

            # GT 2D keypoints - check multiple possible keys
            for key in ['transformed_keypoints', 'keypoints_gt', 'keypoints']:
                if key in gt:
                    gt_kpts_2d = gt[key]
                    if hasattr(gt_kpts_2d, 'cpu'):
                        gt_kpts_2d = gt_kpts_2d.cpu().numpy()
                    break

            # GT 3D keypoints - check multiple possible keys
            for key in ['keypoint3d', 'lifting_target', 'keypoints_gt']:
                if key in gt:
                    gt_kpts_3d = gt[key]
                    if hasattr(gt_kpts_3d, 'cpu'):
                        gt_kpts_3d = gt_kpts_3d.cpu().numpy()
                    break

        # Also check gt_instance_labels for 3D keypoints (used during validation)
        if draw_gt and gt_kpts_3d is None and 'gt_instance_labels' in data_sample:
            gt_labels = data_sample.gt_instance_labels
            if 'keypoint3d' in gt_labels:
                gt_kpts_3d = gt_labels.keypoint3d
                if hasattr(gt_kpts_3d, 'cpu'):
                    gt_kpts_3d = gt_kpts_3d.cpu().numpy()


        # Create visualization
        drawn_img = self._create_visualization(
            image=image,
            pred_kpts_2d=pred_kpts_2d,
            gt_kpts_2d=gt_kpts_2d,
            pred_kpts_3d=pred_kpts_3d,
            gt_kpts_3d=gt_kpts_3d
        )

        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

        return drawn_img
