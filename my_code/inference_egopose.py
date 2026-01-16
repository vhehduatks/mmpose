"""
XR EgoPose Inference Script

This script performs inference using a trained checkpoint and visualizes the results.

Usage:
    python my_code/inference_egopose.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache_test/best_xregopose_Full_Body_All_mpjpe_epoch_7.pth \
        --input F:/ego_cam_dataset/Test/female_001_a_a/env_001/cam_down/rgba/00000200.png \
        --output output_inference/

    # Inference on test dataset samples
    python my_code/inference_egopose.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache_test/best_xregopose_Full_Body_All_mpjpe_epoch_7.pth \
        --dataset-root F:/ego_cam_dataset/Test \
        --num-samples 10 \
        --output output_inference/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# Add mmpose to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmpose.registry import MODELS, DATASETS
from mmpose.structures import PoseDataSample


def parse_args():
    parser = argparse.ArgumentParser(description='XR EgoPose Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image path or folder')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Dataset root to sample from (alternative to --input)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to infer from dataset')
    parser.add_argument('--output', type=str, default='output_inference',
                        help='Output directory for visualization')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization window')
    return parser.parse_args()


class EgoPoseInferencer:
    """Inference class for XR EgoPose models."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """Initialize the inferencer.

        Args:
            config_path: Path to config file
            checkpoint_path: Path to checkpoint file
            device: Device to run inference on
        """
        self.device = device
        self.config = Config.fromfile(config_path)

        # Initialize mmpose scope
        init_default_scope('mmpose')

        # Build model
        self.config.model.train_cfg = None
        self.model = MODELS.build(self.config.model)
        self.model.to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from {checkpoint_path}")

        # Get dataset meta info
        if 'meta' in checkpoint and 'dataset_meta' in checkpoint['meta']:
            self.dataset_meta = checkpoint['meta']['dataset_meta']
        else:
            # Load from config
            self.dataset_meta = self._load_dataset_meta()

        self.model.dataset_meta = self.dataset_meta

        # Build inference pipeline
        self.pipeline = self._build_pipeline()

        # Skeleton info for visualization
        self.skeleton = self._get_skeleton()
        self.kpt_colors = self._get_kpt_colors()
        self.link_colors = self._get_link_colors()

    def _load_dataset_meta(self):
        """Load dataset meta info from config."""
        from mmpose.datasets.datasets.body3d.egopose_info import dataset_info
        from mmpose.datasets.datasets.utils import parse_pose_metainfo

        metainfo = dict(from_file='mmpose/datasets/datasets/body3d/egopose_info.py')
        return parse_pose_metainfo(metainfo)

    def _build_pipeline(self):
        """Build inference pipeline."""
        pipeline_cfg = [
            dict(type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(256, 256), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(47, 47),
                    input_size=(256, 256),
                    sigma=3,
                    type='Custom_mo2cap2_MSRAHeatmap'
                ),
                type='GenerateTarget'
            ),
            dict(type='PackPoseInputs'),
        ]
        return Compose(pipeline_cfg)

    def _get_skeleton(self):
        """Get skeleton connections."""
        # Based on egopose_info.py
        return [
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

    def _get_kpt_colors(self):
        """Get keypoint colors."""
        return np.array([
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

    def _get_link_colors(self):
        """Get skeleton link colors."""
        return np.array([
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

    def preprocess_hmd_info(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """Compute HMD preprocessing info from 3D keypoints.

        The model expects preprocessed HMD info with shape (1, 9):
        - head_dir (3): normalized head direction vector
        - left_hand_dir (3): normalized left hand direction vector
        - right_hand_dir (3): normalized right hand direction vector

        Args:
            keypoints_3d: 3D keypoints with shape (16, 3)

        Returns:
            HMD info with shape (1, 9)
        """
        # Indices: 0=Spine2, 1=Head, 4=LeftHand, 7=RightHand
        spine2 = keypoints_3d[0]
        head = keypoints_3d[1]
        left_hand = keypoints_3d[4]
        right_hand = keypoints_3d[7]

        # Compute direction vectors from spine2 (body center)
        head_dir = head - spine2
        left_hand_dir = left_hand - spine2
        right_hand_dir = right_hand - spine2

        # Normalize
        head_dir = head_dir / (np.linalg.norm(head_dir) + 1e-8)
        left_hand_dir = left_hand_dir / (np.linalg.norm(left_hand_dir) + 1e-8)
        right_hand_dir = right_hand_dir / (np.linalg.norm(right_hand_dir) + 1e-8)

        hmd_info = np.concatenate([head_dir, left_hand_dir, right_hand_dir])
        return hmd_info.reshape(1, 9).astype(np.float32)

    def infer_image(self, img_path: str, hmd_info: np.ndarray = None,
                    keypoints_2d: np.ndarray = None,
                    keypoints_3d: np.ndarray = None) -> dict:
        """Run inference on a single image.

        Args:
            img_path: Path to input image
            hmd_info: Optional HMD info with shape (1, 9). If None, uses dummy values.
            keypoints_2d: Optional 2D keypoints with shape (1, 16, 2). Required for
                          GenerateTarget pipeline. If None, uses center of image.
            keypoints_3d: Optional 3D keypoints with shape (1, 16, 3). Required for
                          Custom codec. If None, uses zeros.

        Returns:
            Dictionary with prediction results
        """
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Prepare dummy keypoints if not provided (needed for GenerateTarget)
        if keypoints_2d is None:
            # Use dummy keypoints at image center
            keypoints_2d = np.full((1, 16, 2), [w/2, h/2], dtype=np.float32)

        if keypoints_3d is None:
            # Use dummy 3D keypoints (zeros)
            keypoints_3d = np.zeros((1, 16, 3), dtype=np.float32)

        # Prepare data
        data_info = {
            'img_path': img_path,
            'img': img_rgb,
            'bbox': np.array([[0, 0, w, h]], dtype=np.float32),
            'bbox_score': np.array([1.0], dtype=np.float32),
            'keypoints': keypoints_2d,
            'keypoints_visible': np.ones((1, 16), dtype=np.float32),
            'keypoint3d': keypoints_3d,
        }
        data_info.update(self.dataset_meta)

        # Apply pipeline
        data = self.pipeline(data_info)

        # Add HMD info to gt_instance_labels
        if hmd_info is None:
            # Use dummy HMD info (zeros)
            hmd_info = np.zeros((1, 9), dtype=np.float32)

        data['data_samples'].gt_instance_labels.set_field(
            torch.from_numpy(hmd_info.astype(np.float32)), 'hmd_info'
        )

        # Create batch
        batch = pseudo_collate([data])

        # Run inference
        with torch.no_grad():
            results = self.model.test_step(batch)

        return {
            'image': img_rgb,
            'pred_instances': results[0].pred_instances,
            'img_path': img_path,
            'gt_keypoints_2d': keypoints_2d,  # Store 2D GT for visualization
        }

    def infer_from_dataset(self, data_root: str, num_samples: int = 5) -> list:
        """Run inference on samples from the dataset.

        This method loads samples from the H5 cache and runs inference with
        proper HMD info.

        Args:
            data_root: Dataset root directory
            num_samples: Number of samples to infer

        Returns:
            List of prediction results
        """
        import h5py

        cache_file = os.path.join(data_root, 'annotations_cache.h5')
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        results = []
        with h5py.File(cache_file, 'r') as hf:
            img_paths = hf['img_paths'][:]
            hmd_infos = hf['hmd_info'][:]
            keypoints_3d_gt = hf['keypoint3d'][:]
            keypoints_2d = hf['keypoints'][:]

            # Sample random indices
            total_samples = len(img_paths)
            indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

            for idx in indices:
                img_path = img_paths[idx].decode('utf-8') if isinstance(img_paths[idx], bytes) else img_paths[idx]
                hmd_info = hmd_infos[idx]  # Shape: (1, 9)
                gt_3d = keypoints_3d_gt[idx]  # Shape: (1, 16, 3)
                kpts_2d = keypoints_2d[idx]  # Shape: (1, 16, 2)

                try:
                    result = self.infer_image(img_path, hmd_info, kpts_2d, gt_3d)
                    result['gt_keypoints_3d'] = gt_3d
                    results.append(result)
                    print(f"Processed: {img_path}")
                except Exception as e:
                    import traceback
                    print(f"Error processing {img_path}: {e}")
                    traceback.print_exc()

        return results

    def _draw_2d_skeleton(self, ax, img: np.ndarray, keypoints: np.ndarray, title: str):
        """Draw 2D skeleton on image.

        Args:
            ax: Matplotlib axis
            img: Image to draw on
            keypoints: (1, 16, 2) or (16, 2) array of 2D keypoints
            title: Plot title
        """
        ax.imshow(img)

        # Handle shape
        if keypoints.ndim == 3:
            keypoints = keypoints[0]  # (16, 2)

        h, w = img.shape[:2]

        # Draw skeleton lines first (so keypoints are on top)
        for idx, (i, j) in enumerate(self.skeleton):
            pt1, pt2 = keypoints[i], keypoints[j]
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color=self.link_colors[idx] / 255.0, linewidth=2)

        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if 0 <= kpt[0] < w and 0 <= kpt[1] < h:
                ax.scatter(kpt[0], kpt[1], c=[self.kpt_colors[i] / 255.0],
                          s=50, marker='o', edgecolors='white', linewidths=1)

        ax.set_title(title)
        ax.axis('off')

    def visualize_result(self, result: dict, output_path: str = None, show: bool = False):
        """Visualize inference result.

        Args:
            result: Inference result dictionary
            output_path: Optional path to save visualization
            show: Whether to display the visualization
        """
        img = result['image']
        pred = result['pred_instances']

        # Get predicted 2D and 3D keypoints
        pred_2d = None
        pred_3d = None
        if hasattr(pred, 'keypoints'):
            pred_2d = pred.keypoints.cpu().numpy() if hasattr(pred.keypoints, 'cpu') else pred.keypoints
        if hasattr(pred, 'keypoint_3d'):
            pred_3d = pred.keypoint_3d.cpu().numpy()[0]  # (16, 3)

        # Get ground truth
        gt_2d = result.get('gt_keypoints_2d', None)
        gt_3d = result.get('gt_keypoints_3d', None)
        if gt_3d is not None:
            gt_3d = gt_3d[0]  # (16, 3)

        # Create figure with 4 panels (2x2)
        fig = plt.figure(figsize=(14, 10))

        # Row 1: 2D visualizations
        # 1. Image + Predicted 2D keypoints
        ax1 = fig.add_subplot(2, 2, 1)
        if pred_2d is not None:
            self._draw_2d_skeleton(ax1, img.copy(), pred_2d, 'Predicted 2D Pose')
        else:
            ax1.imshow(img)
            ax1.set_title('Input Image')
            ax1.axis('off')

        # 2. Image + GT 2D keypoints
        ax2 = fig.add_subplot(2, 2, 2)
        if gt_2d is not None:
            self._draw_2d_skeleton(ax2, img.copy(), gt_2d, 'Ground Truth 2D Pose')
        else:
            ax2.imshow(img)
            ax2.set_title('Input Image (No GT)')
            ax2.axis('off')

        # Row 2: 3D visualizations
        # 3. Predicted 3D pose
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        if pred_3d is not None:
            self._draw_3d_skeleton(ax3, pred_3d, 'Predicted 3D Pose')

        # 4. Ground truth 3D pose
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        if gt_3d is not None:
            self._draw_3d_skeleton(ax4, gt_3d, 'Ground Truth 3D Pose')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")

        if show:
            plt.show()

        plt.close()

    def _draw_3d_skeleton(self, ax, keypoints: np.ndarray, title: str):
        """Draw 3D skeleton on matplotlib axis.

        Args:
            ax: Matplotlib 3D axis
            keypoints: (16, 3) array of 3D keypoints
            title: Plot title
        """
        # Draw keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                   c=self.kpt_colors / 255.0, s=50, marker='o')

        # Draw skeleton
        for idx, (i, j) in enumerate(self.skeleton):
            pts = keypoints[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=self.link_colors[idx] / 255.0, linewidth=2)

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

    def visualize_comparison(self, results: list, output_dir: str):
        """Visualize multiple results in a comparison grid.

        Args:
            results: List of inference results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, result in enumerate(results):
            output_path = os.path.join(output_dir, f'result_{i:04d}.png')
            self.visualize_result(result, output_path)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize inferencer
    print("Initializing model...")
    inferencer = EgoPoseInferencer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    if args.dataset_root:
        # Inference from dataset with proper HMD info
        print(f"Running inference on {args.num_samples} samples from {args.dataset_root}")
        results = inferencer.infer_from_dataset(args.dataset_root, args.num_samples)
        inferencer.visualize_comparison(results, args.output)

    elif args.input:
        # Single image or folder inference
        if os.path.isfile(args.input):
            # Single image
            print(f"Running inference on {args.input}")
            result = inferencer.infer_image(args.input)
            output_path = os.path.join(args.output, 'result.png')
            inferencer.visualize_result(result, output_path, args.show)

        elif os.path.isdir(args.input):
            # Folder of images
            img_extensions = {'.png', '.jpg', '.jpeg'}
            img_files = [f for f in Path(args.input).iterdir()
                         if f.suffix.lower() in img_extensions]

            print(f"Found {len(img_files)} images in {args.input}")
            for img_file in img_files[:args.num_samples]:
                try:
                    result = inferencer.infer_image(str(img_file))
                    output_path = os.path.join(args.output, f'{img_file.stem}_result.png')
                    inferencer.visualize_result(result, output_path, args.show)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    else:
        print("Please specify either --input or --dataset-root")
        return

    print(f"Done! Results saved to {args.output}")


if __name__ == '__main__':
    main()
