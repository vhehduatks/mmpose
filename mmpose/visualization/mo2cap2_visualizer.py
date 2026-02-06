# Copyright (c) OpenMMLab. All rights reserved.
"""
Mo2Cap2 3D Pose Visualizer

Optimized visualizer for Mo2Cap2 dataset with:
- Cached mean3D data loading
- Mo2Cap2-specific skeleton (15 joints)
- Official evaluation protocol visualization
"""
import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmengine.dist import master_only
from mmengine.structures import InstanceData
import scipy.io

from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from . import PoseLocalVisualizer


# Mo2Cap2 constants
MO2CAP2_NUM_JOINTS = 15
MO2CAP2_KINEMATIC_PARENTS = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13])

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


@lru_cache(maxsize=1)
def _load_mo2cap2_bone_lengths():
    """Load mean3D and compute reference bone lengths (cached)."""
    mean3d_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'utils', 'mean3D.mat'
    )
    mean3D = scipy.io.loadmat(mean3d_path)['mean3D']  # (3, 15)
    bones_mean = mean3D - mean3D[:, MO2CAP2_KINEMATIC_PARENTS]
    bone_lengths = np.sqrt(np.sum(bones_mean ** 2, axis=0))  # (15,)
    return mean3D, bone_lengths


def skeleton_rescale(joints, bone_lengths, kinematic_parents):
    """Rescale skeleton to reference bone lengths.

    Args:
        joints: (3, 15) joint positions
        bone_lengths: (14,) target bone lengths (excluding root)
        kinematic_parents: (15,) parent indices

    Returns:
        (3, 15) rescaled joint positions
    """
    rescaled = np.zeros_like(joints)
    rescaled[:, 0] = joints[:, 0]  # Root stays same

    for i in range(1, MO2CAP2_NUM_JOINTS):
        parent = kinematic_parents[i]
        bone = joints[:, i] - joints[:, parent]
        bone_norm = np.linalg.norm(bone)
        if bone_norm > 1e-8:
            bone = bone * bone_lengths[i - 1] / bone_norm
        rescaled[:, i] = rescaled[:, parent] + bone

    return rescaled


def procrustes_align(target, source, scaling=False):
    """Procrustes alignment: align source to target.

    Args:
        target: (N, 3) target coordinates
        source: (N, 3) source coordinates to transform
        scaling: whether to allow scaling

    Returns:
        (N, 3) aligned source coordinates
    """
    mu_target = target.mean(0)
    mu_source = source.mean(0)

    target_centered = target - mu_target
    source_centered = source - mu_source

    norm_target = np.linalg.norm(target_centered)
    norm_source = np.linalg.norm(source_centered)

    target_norm = target_centered / norm_target
    source_norm = source_centered / norm_source

    H = target_norm.T @ source_norm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if scaling:
        scale = S.sum() * norm_target / norm_source
        aligned = norm_target * S.sum() * (source_norm @ R) + mu_target
    else:
        aligned = norm_source * (source_norm @ R) + mu_target

    return aligned


@VISUALIZERS.register_module()
class Mo2Cap2Visualizer(PoseLocalVisualizer):
    """Optimized 3D pose visualizer for Mo2Cap2 dataset.

    Features:
    - Cached bone length loading
    - Mo2Cap2-specific skeleton visualization
    - Official evaluation protocol (rescale + Procrustes)

    Args:
        name (str): Visualizer name. Defaults to 'visualizer'.
        vis_backends (list, optional): Visualization backends.
        save_dir (str, optional): Save directory.
        line_width (int): Line width for skeleton. Defaults to 2.
        radius (int): Keypoint radius. Defaults to 4.
        alpha (float): Transparency. Defaults to 0.8.
    """

    def __init__(
        self,
        name: str = 'visualizer',
        image: Optional[np.ndarray] = None,
        vis_backends: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        skeleton: Optional[Union[List, Tuple]] = None,
        line_width: Union[int, float] = 2,
        radius: Union[int, float] = 4,
        show_keypoint_weight: bool = False,
        backend: str = 'opencv',
        alpha: float = 0.8,
    ):
        super().__init__(
            name, image, vis_backends, save_dir, bbox_color,
            kpt_color, link_color, text_color, skeleton,
            line_width, radius, show_keypoint_weight, backend, alpha
        )
        # Pre-load bone lengths
        self._mean3D, self._bone_lengths = _load_mo2cap2_bone_lengths()

    def _draw_3d_pose(
        self,
        ax,
        keypoints: np.ndarray,
        color: str = 'blue',
        title: str = '',
        show_joints: bool = True,
        show_skeleton: bool = True,
    ):
        """Draw 3D pose on matplotlib axis.

        Args:
            ax: Matplotlib 3D axis
            keypoints: (15, 3) joint positions
            color: Color for joints and skeleton
            title: Plot title
            show_joints: Whether to draw joint markers
            show_skeleton: Whether to draw skeleton lines
        """
        if show_joints:
            ax.scatter(
                keypoints[:, 0],
                keypoints[:, 1],
                keypoints[:, 2],
                c=color, s=30, marker='o'
            )

        if show_skeleton:
            for (i, j) in MO2CAP2_SKELETON:
                ax.plot(
                    [keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    [keypoints[i, 2], keypoints[j, 2]],
                    c=color, linewidth=self.line_width
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            ax.set_title(title)

    def _draw_3d_comparison(
        self,
        pred_kpts: np.ndarray,
        gt_kpts: np.ndarray,
        figsize: Tuple[int, int] = (12, 5),
        elev: float = 15,
        azim: float = 70,
    ) -> np.ndarray:
        """Draw side-by-side 3D comparison of prediction and GT.

        Args:
            pred_kpts: (15, 3) predicted keypoints
            gt_kpts: (15, 3) ground truth keypoints
            figsize: Figure size
            elev: Elevation angle
            azim: Azimuth angle

        Returns:
            np.ndarray: RGB image of the visualization
        """
        # Rescale to reference bone lengths
        pred_rescaled = skeleton_rescale(
            pred_kpts.T, self._bone_lengths[1:], MO2CAP2_KINEMATIC_PARENTS
        ).T
        gt_rescaled = skeleton_rescale(
            gt_kpts.T, self._bone_lengths[1:], MO2CAP2_KINEMATIC_PARENTS
        ).T

        # Align prediction to GT using Procrustes
        pred_aligned = procrustes_align(gt_rescaled, pred_rescaled, scaling=False)

        plt.ioff()
        fig = plt.figure(figsize=figsize)

        # Prediction plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.view_init(elev=elev, azim=azim)
        self._draw_3d_pose(ax1, pred_aligned, color='blue', title='Prediction')

        # Ground truth plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.view_init(elev=elev, azim=azim)
        self._draw_3d_pose(ax2, gt_rescaled, color='green', title='Ground Truth')

        fig.tight_layout()
        fig.canvas.draw()

        # Convert to numpy array
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img_data = img_data.reshape(int(height), int(width), 3)

        plt.close(fig)
        return img_data

    def _draw_2d_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        kpt_thr: float = 0.3,
        is_gt: bool = False,
    ) -> np.ndarray:
        """Draw 2D keypoints on image.

        Args:
            image: Input image (H, W, 3)
            keypoints: (N, 2) or (N, 3) keypoint coordinates
            scores: (N,) keypoint scores
            kpt_thr: Score threshold
            is_gt: If True, use GT colors (cyan/magenta), else prediction colors (green/red)

        Returns:
            Image with keypoints drawn
        """
        img = image.copy()

        if scores is None:
            scores = np.ones(len(keypoints))

        kpts = keypoints[:, :2].astype(np.int32)

        # Color scheme: GT uses cyan/magenta, Pred uses green/red
        if is_gt:
            upper_color = (255, 255, 0)  # Cyan (BGR)
            lower_color = (255, 0, 255)  # Magenta (BGR)
            upper_kpt_color = (255, 255, 0)
            lower_kpt_color = (255, 0, 255)
        else:
            upper_color = (0, 200, 0)  # Green
            lower_color = (0, 0, 200)  # Red
            upper_kpt_color = (0, 255, 0)
            lower_kpt_color = (0, 0, 255)

        # Helper to check if coordinate is valid (within image bounds)
        h, w = img.shape[:2]
        def is_valid_coord(x, y):
            return 0 <= x < w and 0 <= y < h

        # Draw skeleton first (so keypoints are on top)
        for (i, j) in MO2CAP2_SKELETON:
            if scores[i] < kpt_thr or scores[j] < kpt_thr:
                continue
            pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            # Skip if either endpoint is invalid
            if not is_valid_coord(pt1[0], pt1[1]) or not is_valid_coord(pt2[0], pt2[1]):
                continue
            color = upper_color if i < 7 and j < 7 else lower_color
            cv2.line(img, pt1, pt2, color, int(self.line_width))

        # Draw keypoints
        for i, (kpt, score) in enumerate(zip(kpts, scores)):
            if score < kpt_thr:
                continue
            center = (int(kpt[0]), int(kpt[1]))
            # Skip if coordinate is invalid
            if not is_valid_coord(center[0], center[1]):
                continue
            color = upper_kpt_color if i < 7 else lower_kpt_color
            cv2.circle(img, center, int(self.radius), color, -1)

        return img

    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: PoseDataSample,
        det_data_sample: Optional[PoseDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        draw_2d: bool = True,
        draw_bbox: bool = False,
        show_kpt_idx: bool = False,
        skeleton_style: str = 'mmpose',
        dataset_2d: str = 'coco',
        dataset_3d: str = 'mo2cap2',
        convert_keypoint: bool = False,
        axis_azimuth: float = 70,
        axis_limit: float = 1.7,
        axis_dist: float = 10.0,
        axis_elev: float = 15.0,
        num_instances: int = -1,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        kpt_thr: float = 0.0,  # Mo2Cap2: use 0.0 since heatmap scores are low
        step: int = 0,
    ) -> np.ndarray:
        """Visualize Mo2Cap2 pose estimation results.

        Handles two cases:
        - Training: Has both 2D GT (cyan/magenta) and 2D predictions (green/red)
        - Validation: Only has 2D predictions (green/red), no 2D GT

        Args:
            name: Image identifier
            image: Input image (H, W, 3)
            data_sample: Pose data sample with predictions and GT
            draw_gt: Whether to draw ground truth
            draw_pred: Whether to draw predictions
            draw_2d: Whether to draw 2D keypoints
            axis_azimuth: 3D view azimuth angle
            axis_elev: 3D view elevation angle
            show: Whether to display
            out_file: Output file path
            kpt_thr: Keypoint score threshold
            step: Global step for logging

        Returns:
            Visualization image
        """
        vis_parts = []

        # Get target size from data_sample metainfo if available
        target_size = (256, 256)  # Default Mo2Cap2 input size
        if hasattr(data_sample, 'metainfo'):
            input_size = data_sample.metainfo.get('input_size', target_size)
            if input_size is not None:
                target_size = tuple(input_size)

        # For validation images with disk paths, always prefer loading from disk
        # This ensures correct visualization regardless of denormalization issues
        img_path = None
        if hasattr(data_sample, 'metainfo'):
            img_path = data_sample.metainfo.get('img_path')

        # If we have a valid disk path (not H5), load from disk for reliability
        if img_path is not None and not img_path.startswith('h5://') and os.path.exists(img_path):
            try:
                loaded_img = cv2.imread(img_path)
                if loaded_img is not None:
                    image = loaded_img
            except Exception:
                pass  # Keep denormalized image if loading fails

        # Fallback check for H5 images: detect obviously invalid denormalization
        img_mean = image.mean()
        img_std = image.std() if len(image.shape) > 0 else 0
        is_invalid = img_mean > 250 or img_mean < 5 or img_std < 5

        if is_invalid and img_path is not None and not img_path.startswith('h5://'):
            # Try to load from disk as last resort
            try:
                loaded_img = cv2.imread(img_path)
                if loaded_img is not None:
                    image = loaded_img
            except Exception:
                pass

        # Resize image to match keypoint coordinate space if needed
        # Mo2Cap2 val/test images are loaded at original size (e.g., 1024x1280)
        # but keypoints are predicted in 256x256 space after TopdownAffine
        if image.shape[0] != target_size[1] or image.shape[1] != target_size[0]:
            # Apply same transforms as validation pipeline:
            # 1. CenterCrop: 1280x1024 -> 1024x1024 (remove 128px margins)
            # 2. Resize to 256x256
            h, w = image.shape[:2]

            # Mo2Cap2 CenterCrop parameters (same as val pipeline)
            margin_left = 128
            margin_right = 128

            # Only apply center crop if image is wider than tall (Mo2Cap2 test images)
            if w > h and w - margin_left - margin_right > 0:
                image = image[:, margin_left:w-margin_right].copy()

            # Resize to target size
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # 1. Draw 2D keypoints on image
        if draw_2d:
            img_2d = image.copy()
            has_2d_vis = False
            gt_2d = None  # Initialize for legend check

            # Draw GT 2D keypoints first (if available - training only)
            if draw_gt:
                # Try gt_instances first, then gt_instance_labels
                if 'gt_instances' in data_sample:
                    gt_inst = data_sample.gt_instances
                    if 'keypoints' in gt_inst:
                        gt_2d = gt_inst.keypoints
                elif 'gt_instance_labels' in data_sample:
                    gt_inst = data_sample.gt_instance_labels
                    if 'keypoints' in gt_inst:
                        gt_2d = gt_inst.keypoints

                if gt_2d is not None:
                    if hasattr(gt_2d, 'cpu'):
                        gt_2d = gt_2d.cpu().numpy()
                    gt_2d = np.array(gt_2d)
                    if gt_2d.ndim == 3:
                        gt_2d = gt_2d[0]  # (15, 2)
                    # Check if GT keypoints are valid (not all zeros)
                    # Test data has no 2D GT, so keypoints are zeros
                    if np.abs(gt_2d).max() > 1e-6:
                        # Draw GT with cyan/magenta colors
                        img_2d = self._draw_2d_keypoints(
                            img_2d, gt_2d, scores=None, kpt_thr=0.0, is_gt=True
                        )
                        has_2d_vis = True
                    else:
                        gt_2d = None  # Mark as invalid for legend

            # Draw predicted 2D keypoints (green/red)
            if draw_pred and 'pred_instances' in data_sample:
                pred_inst = data_sample.pred_instances
                if 'keypoints' in pred_inst:
                    kpts_2d = pred_inst.keypoints
                    if hasattr(kpts_2d, 'cpu'):
                        kpts_2d = kpts_2d.cpu().numpy()
                    kpts_2d = np.array(kpts_2d).copy()
                    if kpts_2d.ndim == 3:
                        kpts_2d = kpts_2d[0]  # (15, 2 or 3)

                    # Transform keypoints from original image space back to 256x256 visualization space
                    # TopdownPoseEstimator applies: kp_orig = (kp_256 / input_size * input_scale) + input_center - 0.5 * input_scale
                    # Reverse: kp_256 = (kp_orig - input_center + 0.5 * input_scale) * input_size / input_scale
                    if hasattr(data_sample, 'metainfo'):
                        meta = data_sample.metainfo
                        input_center = meta.get('input_center', None)
                        input_scale = meta.get('input_scale', None)
                        input_size = meta.get('input_size', target_size)

                        if input_center is not None and input_scale is not None:
                            input_center = np.array(input_center)
                            input_scale = np.array(input_scale)
                            input_size = np.array(input_size)
                            # Reverse the transformation
                            kpts_2d[:, :2] = (kpts_2d[:, :2] - input_center + 0.5 * input_scale) * input_size / input_scale

                    scores = pred_inst.get('keypoint_scores', np.ones(len(kpts_2d)))
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    scores = np.array(scores)
                    if scores.ndim > 1:
                        scores = scores[0]

                    img_2d = self._draw_2d_keypoints(
                        img_2d, kpts_2d, scores, kpt_thr, is_gt=False
                    )
                    has_2d_vis = True

            # Always add 2D image panel (even if no keypoints are drawn)
            # Add legend text
            legend_y = 20
            cv2.putText(img_2d, 'Pred: Green/Red', (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if gt_2d is not None:
                cv2.putText(img_2d, 'GT: Cyan/Magenta', (10, legend_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            vis_parts.append(img_2d)

        # 2. Draw 3D pose comparison
        pred_3d = None
        gt_3d = None

        if draw_pred and 'pred_instances' in data_sample:
            pred_inst = data_sample.pred_instances
            if 'keypoint_3d' in pred_inst:
                pred_3d = pred_inst.keypoint_3d
                if hasattr(pred_3d, 'cpu'):
                    pred_3d = pred_3d.cpu().numpy()
                pred_3d = pred_3d.reshape(MO2CAP2_NUM_JOINTS, 3)

        if draw_gt:
            gt_3d = None
            # Try gt_instances first, then gt_instance_labels (Mo2Cap2 uses this)
            if 'gt_instances' in data_sample:
                gt_inst = data_sample.gt_instances
                if 'keypoint3d' in gt_inst:
                    gt_3d = gt_inst.keypoint3d
                elif 'lifting_target' in gt_inst:
                    gt_3d = gt_inst.lifting_target
            elif 'gt_instance_labels' in data_sample:
                gt_inst = data_sample.gt_instance_labels
                if 'keypoint3d' in gt_inst:
                    gt_3d = gt_inst.keypoint3d

            if gt_3d is not None:
                if hasattr(gt_3d, 'cpu'):
                    gt_3d = gt_3d.cpu().numpy()
                gt_3d = np.array(gt_3d).reshape(MO2CAP2_NUM_JOINTS, 3)

        if pred_3d is not None and gt_3d is not None:
            img_3d = self._draw_3d_comparison(
                pred_3d, gt_3d,
                elev=axis_elev, azim=axis_azimuth
            )
            # Convert RGB (from matplotlib) to BGR (for consistency with 2D image)
            img_3d = img_3d[:, :, ::-1].copy()
            vis_parts.append(img_3d)

        # Combine visualizations
        if len(vis_parts) == 0:
            drawn_img = image
        elif len(vis_parts) == 1:
            drawn_img = vis_parts[0]
        else:
            # Resize to same height and concatenate
            max_h = max(p.shape[0] for p in vis_parts)
            resized = []
            for p in vis_parts:
                if p.shape[0] < max_h:
                    pad_h = max_h - p.shape[0]
                    p = cv2.copyMakeBorder(
                        p, pad_h // 2, pad_h - pad_h // 2, 0, 0,
                        cv2.BORDER_CONSTANT, value=(255, 255, 255)
                    )
                resized.append(p)
            drawn_img = np.concatenate(resized, axis=1)

        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # Convert BGR to RGB for WandB/visualization backends
            # (OpenCV uses BGR, but WandB expects RGB)
            drawn_img_rgb = drawn_img[:, :, ::-1].copy()
            self.add_image(name, drawn_img_rgb, step)

        return self.get_image()
