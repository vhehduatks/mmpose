# Copyright (c) OpenMMLab. All rights reserved.
"""Simplified 3D Pose Visualizer for XR EgoPose dataset."""

import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmengine.dist import master_only
from mmengine.structures import InstanceData

from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from . import PoseLocalVisualizer


@VISUALIZERS.register_module()
class CustomPose3dLocalVisualizer_xregopose_v2(PoseLocalVisualizer):
    """Simplified 3D Pose Visualizer for XR EgoPose.

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

    def _get_color_array(self, color, length: int) -> np.ndarray:
        """Convert color to numpy array of shape (length, 3)."""
        if color is None or isinstance(color, str):
            return np.array([[255, 0, 0]] * length)  # default red
        color = np.array(color)
        if color.ndim == 1:
            return np.tile(color, (length, 1))
        return color[:length]

    def _draw_3d_skeleton(self,
                          ax,
                          keypoints: np.ndarray,
                          scores: np.ndarray,
                          kpt_thr: float = 0.3,
                          title: str = None):
        """Draw 3D skeleton on matplotlib axis.

        Args:
            ax: Matplotlib 3D axis
            keypoints: (N, 3) array of 3D keypoints
            scores: (N,) array of keypoint scores
            kpt_thr: Score threshold for visibility
            title: Plot title
        """
        valid = (scores >= kpt_thr) & (~np.isnan(keypoints).any(axis=-1))
        kpts = keypoints[valid]

        if len(kpts) == 0:
            return

        # Set axis properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            ax.set_title(title, y=1.1)

        # Set axis limits centered on keypoints
        center = kpts.mean(axis=0)
        limit = 1.0
        ax.set_xlim3d([center[0] - limit/2, center[0] + limit/2])
        ax.set_ylim3d([center[1] - limit/2, center[1] + limit/2])
        ax.set_zlim3d([center[2] + limit/2, max(0, center[2] - limit/2)])
        ax.set_box_aspect([1, 1, 1])

        # Draw keypoints
        colors = self._get_color_array(self.kpt_color, len(keypoints))
        colors_valid = colors[valid] / 255.0
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c=colors_valid, marker='o')

        # Draw skeleton
        if self.skeleton and self.link_color is not None:
            link_colors = self._get_color_array(self.link_color, len(self.skeleton))
            for sk_id, (i, j) in enumerate(self.skeleton):
                if scores[i] >= kpt_thr and scores[j] >= kpt_thr:
                    pts = keypoints[[i, j]]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                           color=link_colors[sk_id] / 255.0)

    def _draw_3d_comparison(self,
                            pred_kpts: np.ndarray,
                            gt_kpts: Optional[np.ndarray] = None,
                            kpt_thr: float = 0.3) -> np.ndarray:
        """Draw 3D prediction and optionally ground truth side by side.

        Args:
            pred_kpts: (1, N, 3) predicted keypoints
            gt_kpts: (1, N, 3) ground truth keypoints, optional
            kpt_thr: Score threshold

        Returns:
            np.ndarray: Rendered image as RGB array
        """
        num_plots = 2 if gt_kpts is not None else 1

        plt.ioff()
        fig = plt.figure(figsize=(5 * num_plots, 5))

        # Draw prediction
        ax_pred = fig.add_subplot(1, num_plots, 1, projection='3d')
        ax_pred.view_init(elev=15, azim=70)
        scores = np.ones(pred_kpts.shape[1])
        self._draw_3d_skeleton(ax_pred, pred_kpts[0], scores, kpt_thr, 'Prediction')

        # Draw ground truth if available
        if gt_kpts is not None:
            ax_gt = fig.add_subplot(1, num_plots, 2, projection='3d')
            ax_gt.view_init(elev=15, azim=70)
            self._draw_3d_skeleton(ax_gt, gt_kpts[0], scores, kpt_thr, 'Ground Truth')

        # Convert to image
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.get_size_inches() * fig.get_dpi()
        img = img.reshape(int(h), int(w), 3)
        plt.close(fig)

        return img

    def _draw_2d_keypoints(self,
                           image: np.ndarray,
                           keypoints: np.ndarray,
                           scores: np.ndarray,
                           kpt_thr: float = 0.3) -> np.ndarray:
        """Draw 2D keypoints on image.

        Args:
            image: Input image (H, W, 3)
            keypoints: (N, 2) keypoint coordinates
            scores: (N,) keypoint scores
            kpt_thr: Score threshold

        Returns:
            np.ndarray: Image with keypoints drawn
        """
        self.set_image(image)
        h, w = image.shape[:2]

        colors = self._get_color_array(self.kpt_color, len(keypoints))

        # Draw keypoints
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < kpt_thr:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < w and 0 <= y < h:
                self.draw_circles(
                    kpt[:2],
                    radius=np.array([self.radius]),
                    face_colors=tuple(colors[i].tolist()),
                    edge_colors=tuple(colors[i].tolist()),
                    alpha=self.alpha,
                    line_widths=self.radius
                )

        # Draw skeleton
        if self.skeleton and self.link_color is not None:
            link_colors = self._get_color_array(self.link_color, len(self.skeleton))
            for sk_id, (i, j) in enumerate(self.skeleton):
                if scores[i] >= kpt_thr and scores[j] >= kpt_thr:
                    pt1, pt2 = keypoints[i], keypoints[j]
                    if all(0 <= pt1[k] < [w, h][k] and 0 <= pt2[k] < [w, h][k] for k in [0, 1]):
                        self.draw_lines(
                            np.array([pt1[0], pt2[0]]),
                            np.array([pt1[1], pt2[1]]),
                            tuple(link_colors[sk_id].tolist()),
                            line_widths=self.line_width
                        )

        return self.get_image()

    def _merge_images(self, img_2d: np.ndarray, img_3d: np.ndarray) -> np.ndarray:
        """Merge 2D and 3D visualization images horizontally.

        Args:
            img_2d: 2D keypoint image
            img_3d: 3D skeleton image

        Returns:
            np.ndarray: Merged image
        """
        # Match heights
        h1, h2 = img_2d.shape[0], img_3d.shape[0]
        if h1 != h2:
            target_h = max(h1, h2)
            if h1 < target_h:
                pad = (target_h - h1) // 2
                img_2d = cv2.copyMakeBorder(img_2d, pad, target_h - h1 - pad,
                                            0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if h2 < target_h:
                pad = (target_h - h2) // 2
                img_3d = cv2.copyMakeBorder(img_3d, pad, target_h - h2 - pad,
                                            0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Add margin to 2D image
        margin = 50
        img_2d = cv2.copyMakeBorder(img_2d, 0, 0, margin, margin,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))

        return np.concatenate([img_2d, img_3d], axis=1)

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
        img_2d = None
        pred_kpts_3d = None
        gt_kpts_3d = None

        # Extract prediction 3D keypoints
        if draw_pred and 'pred_instances' in data_sample:
            pred = data_sample.pred_instances
            if 'keypoint_3d' in pred:
                pred_kpts_3d = pred.keypoint_3d.cpu().numpy()

            # Draw 2D keypoints
            if draw_2d and 'keypoints' in pred:
                kpts_2d = pred.get('transformed_keypoints', pred.keypoints)[0]
                scores_2d = pred.get('keypoint_scores', np.ones(len(kpts_2d)))[0]
                img_2d = self._draw_2d_keypoints(image.copy(), kpts_2d, scores_2d, kpt_thr)

        # Extract ground truth 3D keypoints
        if draw_gt and 'gt_instances' in data_sample:
            gt = data_sample.gt_instances
            for key in ['keypoint3d', 'lifting_target', 'keypoints_gt']:
                if key in gt:
                    gt_kpts_3d = gt[key]
                    break

        # Draw 3D visualization
        if pred_kpts_3d is not None:
            img_3d = self._draw_3d_comparison(
                pred_kpts_3d,
                gt_kpts_3d if draw_gt else None,
                kpt_thr
            )
        else:
            img_3d = np.full((256, 256, 3), 255, dtype=np.uint8)

        # Merge images
        if img_2d is not None:
            drawn_img = self._merge_images(img_2d, img_3d)
        else:
            drawn_img = img_3d

        self.set_image(drawn_img)

        # Output handling
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

        return drawn_img
