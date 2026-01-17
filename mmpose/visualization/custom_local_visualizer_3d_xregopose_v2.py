# Copyright (c) OpenMMLab. All rights reserved.
"""Simplified 3D Pose Visualizer for XR EgoPose dataset."""

import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
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

# EgoPose keypoint colors (RGB format)
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
])

# EgoPose skeleton link colors
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
])


@VISUALIZERS.register_module()
class CustomPose3dLocalVisualizer_xregopose_v2(PoseLocalVisualizer):
    """Simplified 3D Pose Visualizer for XR EgoPose.

    Visualizes:
    - Row 1: Predicted 2D pose, Ground Truth 2D pose
    - Row 2: Predicted 3D pose, Ground Truth 3D pose
    - Row 3: Predicted heatmaps (sum)

    Args:
        name (str): Name of visualizer instance
        image (np.ndarray, optional): Origin image to draw
        vis_backends (list, optional): Visual backend config list
        save_dir (str, optional): Save file directory
        line_width (int): Width of skeleton lines. Default: 4
        radius (int): Radius of keypoints. Default: 3
        alpha (float): Transparency. Default: 0.8
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: int = 4,
                 radius: int = 3,
                 alpha: float = 0.8,
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
        # Use EgoPose-specific skeleton and colors as defaults
        self._egopose_skeleton = EGOPOSE_SKELETON
        self._egopose_kpt_colors = EGOPOSE_KPT_COLORS / 255.0  # Normalized for matplotlib
        self._egopose_link_colors = EGOPOSE_LINK_COLORS / 255.0

    def _draw_2d_skeleton_plt(self, ax, image: np.ndarray, keypoints: np.ndarray,
                               title: str = '2D Pose'):
        """Draw 2D skeleton on matplotlib axis.

        Args:
            ax: Matplotlib axis
            image: RGB image (H, W, 3)
            keypoints: (N, 2) or (1, N, 2) array of 2D keypoints
            title: Plot title
        """
        ax.imshow(image)

        if keypoints is None:
            ax.set_title(title)
            ax.axis('off')
            return

        if keypoints.ndim == 3:
            keypoints = keypoints[0]

        h, w = image.shape[:2]

        # Draw skeleton lines first
        for idx, (i, j) in enumerate(self._egopose_skeleton):
            if i < len(keypoints) and j < len(keypoints):
                pt1, pt2 = keypoints[i], keypoints[j]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                            color=self._egopose_link_colors[idx], linewidth=2)

        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if i < len(self._egopose_kpt_colors):
                if 0 <= kpt[0] < w and 0 <= kpt[1] < h:
                    ax.scatter(kpt[0], kpt[1], c=[self._egopose_kpt_colors[i]],
                              s=50, marker='o', edgecolors='white', linewidths=1, zorder=5)

        ax.set_title(title)
        ax.axis('off')

    def _draw_3d_skeleton_plt(self, ax, keypoints: np.ndarray, title: str = '3D Pose'):
        """Draw 3D skeleton on matplotlib axis.

        Args:
            ax: Matplotlib 3D axis
            keypoints: (N, 3) or (1, N, 3) array of 3D keypoints
            title: Plot title
        """
        if keypoints is None:
            ax.set_title(title)
            return

        if keypoints.ndim == 3:
            keypoints = keypoints[0]

        # Draw keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                   c=self._egopose_kpt_colors, s=50, marker='o')

        # Draw skeleton
        for idx, (i, j) in enumerate(self._egopose_skeleton):
            if i < len(keypoints) and j < len(keypoints):
                pts = keypoints[[i, j]]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=self._egopose_link_colors[idx], linewidth=2)

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

    def _draw_heatmaps_plt(self, ax, heatmaps: np.ndarray, title: str = 'Predicted Heatmaps'):
        """Draw sum of heatmaps on matplotlib axis.

        Args:
            ax: Matplotlib axis
            heatmaps: (C, H, W) array of heatmaps
            title: Plot title
        """
        if heatmaps is None:
            ax.set_title(title)
            ax.axis('off')
            return

        if hasattr(heatmaps, 'cpu'):
            heatmaps = heatmaps.cpu().numpy()

        # Sum all channels
        heatmap_sum = heatmaps.sum(axis=0)

        ax.imshow(heatmap_sum, cmap='jet')
        ax.set_title(title)
        ax.axis('off')

    def _create_visualization(self,
                               image: np.ndarray,
                               pred_kpts_2d: Optional[np.ndarray] = None,
                               gt_kpts_2d: Optional[np.ndarray] = None,
                               pred_kpts_3d: Optional[np.ndarray] = None,
                               gt_kpts_3d: Optional[np.ndarray] = None,
                               pred_heatmaps: Optional[np.ndarray] = None) -> np.ndarray:
        """Create full visualization with 2D, 3D poses and heatmaps.

        Layout:
        - Row 1: Predicted 2D | GT 2D
        - Row 2: Predicted 3D | GT 3D
        - Row 3: Heatmaps (spanning both columns)

        Args:
            image: Input image (H, W, 3) in RGB
            pred_kpts_2d: Predicted 2D keypoints
            gt_kpts_2d: Ground truth 2D keypoints
            pred_kpts_3d: Predicted 3D keypoints
            gt_kpts_3d: Ground truth 3D keypoints
            pred_heatmaps: Predicted heatmaps (C, H, W)

        Returns:
            np.ndarray: Rendered visualization as RGB image
        """
        plt.ioff()

        # Determine layout based on available data
        has_heatmaps = pred_heatmaps is not None
        num_rows = 3 if has_heatmaps else 2

        fig = plt.figure(figsize=(10, 5 * num_rows))
        gs = gridspec.GridSpec(num_rows, 2, figure=fig, hspace=0.3, wspace=0.2)

        # Row 1: 2D poses
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_2d_skeleton_plt(ax1, image, pred_kpts_2d, 'Predicted 2D Pose')

        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_2d_skeleton_plt(ax2, image, gt_kpts_2d, 'Ground Truth 2D Pose')

        # Row 2: 3D poses
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        self._draw_3d_skeleton_plt(ax3, pred_kpts_3d, 'Predicted 3D Pose')

        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        self._draw_3d_skeleton_plt(ax4, gt_kpts_3d, 'Ground Truth 3D Pose')

        # Row 3: Heatmaps
        if has_heatmaps:
            ax5 = fig.add_subplot(gs[2, :])
            self._draw_heatmaps_plt(ax5, pred_heatmaps, 'Predicted Heatmaps (Sum)')

        # Convert to image
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
        """Draw datasample and save to backends.

        Creates visualization with:
        - Predicted 2D pose | Ground Truth 2D pose
        - Predicted 3D pose | Ground Truth 3D pose
        - Predicted heatmaps

        Args:
            name: Image identifier
            image: Input image (H, W, 3) in RGB
            data_sample: Pose data sample with predictions/ground truth
            draw_gt: Whether to draw ground truth
            draw_pred: Whether to draw predictions
            draw_2d: Whether to draw 2D keypoints
            show: Whether to display image
            wait_time: Display wait time
            out_file: Output file path
            kpt_thr: Keypoint score threshold
            step: Global step for logging

        Returns:
            np.ndarray: Drawn image
        """
        pred_kpts_2d = None
        gt_kpts_2d = None
        pred_kpts_3d = None
        gt_kpts_3d = None
        pred_heatmaps = None

        # Extract prediction data
        if draw_pred and 'pred_instances' in data_sample:
            pred = data_sample.pred_instances

            # 2D keypoints
            if 'keypoints' in pred:
                kpts = pred.get('transformed_keypoints', pred.keypoints)
                if hasattr(kpts, 'cpu'):
                    kpts = kpts.cpu().numpy()
                pred_kpts_2d = kpts

            # 3D keypoints
            if 'keypoint_3d' in pred:
                kpts_3d = pred.keypoint_3d
                if hasattr(kpts_3d, 'cpu'):
                    kpts_3d = kpts_3d.cpu().numpy()
                pred_kpts_3d = kpts_3d

        # Extract predicted heatmaps from pred_fields
        if draw_pred and 'pred_fields' in data_sample:
            pred_fields = data_sample.pred_fields
            if 'heatmaps' in pred_fields:
                pred_heatmaps = pred_fields.heatmaps
                if hasattr(pred_heatmaps, 'cpu'):
                    pred_heatmaps = pred_heatmaps.cpu().numpy()

        # Extract ground truth data
        if draw_gt and 'gt_instances' in data_sample:
            gt = data_sample.gt_instances

            # 2D keypoints (transformed)
            if 'transformed_keypoints' in gt:
                gt_kpts_2d = gt.transformed_keypoints
            elif 'keypoints' in gt:
                gt_kpts_2d = gt.keypoints

            if hasattr(gt_kpts_2d, 'cpu'):
                gt_kpts_2d = gt_kpts_2d.cpu().numpy()

            # 3D keypoints
            for key in ['keypoint3d', 'lifting_target', 'keypoints_gt']:
                if key in gt:
                    gt_kpts_3d = gt[key]
                    if hasattr(gt_kpts_3d, 'cpu'):
                        gt_kpts_3d = gt_kpts_3d.cpu().numpy()
                    break

        # Create visualization
        drawn_img = self._create_visualization(
            image=image,
            pred_kpts_2d=pred_kpts_2d,
            gt_kpts_2d=gt_kpts_2d,
            pred_kpts_3d=pred_kpts_3d,
            gt_kpts_3d=gt_kpts_3d,
            pred_heatmaps=pred_heatmaps
        )

        self.set_image(drawn_img)

        # Output handling
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

        return drawn_img
