"""
Inference Script for FC-exported EgoPose ONNX Model

This script runs inference using the properly exported ONNX model
that includes the full FC layers (encoder + pose_decoder).

Usage:
    python my_code/deploy/inference_egopose_fc.py \
        --onnx deploy_models/egopose_fc/end2end.onnx \
        --image F:/ego_cam_dataset/Train/female_002_a_a/env_001/cam_down/rgba/female_002_a_a.rgba.000001.png
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def preprocess_image(image_path, input_size=(256, 256)):
    """Preprocess image for inference.

    Args:
        image_path: Path to input image
        input_size: Model input size (H, W)

    Returns:
        input_tensor: (1, 3, H, W) float32 array
        original_image: Original BGR image for visualization
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    original_image = image.copy()

    # Handle RGBA images
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Resize
    image = cv2.resize(image, input_size)

    # Normalize (ImageNet mean/std)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    image = image.astype(np.float32)
    image = (image - mean) / std

    # HWC -> CHW -> NCHW
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32), original_image


def run_inference(onnx_path, input_tensor):
    """Run ONNX inference.

    Args:
        onnx_path: Path to ONNX model
        input_tensor: (1, 3, H, W) input

    Returns:
        heatmaps: (1, 16, 32, 32)
        pose_3d: (1, 16, 3)
        latency_ms: Inference time in milliseconds
    """
    import onnxruntime as ort

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Input: {input_name}")
    print(f"Outputs: {output_names}")

    # Warmup
    for _ in range(3):
        session.run(output_names, {input_name: input_tensor})

    # Benchmark
    num_runs = 10
    start = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(output_names, {input_name: input_tensor})
    end = time.perf_counter()

    latency_ms = (end - start) / num_runs * 1000

    # Parse outputs
    heatmaps = outputs[0]  # (1, 16, 32, 32)
    pose_3d = outputs[1]   # (1, 16, 3)

    return heatmaps, pose_3d, latency_ms


def visualize_results(original_image, heatmaps, pose_3d, output_path=None):
    """Visualize inference results.

    Args:
        original_image: Original BGR image
        heatmaps: (1, 16, 32, 32) heatmap predictions
        pose_3d: (1, 16, 3) 3D pose predictions
        output_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # EgoPose joint names
    joint_names = [
        'head', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist',
        'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle',
        'left_foot', 'right_foot'
    ]

    # EgoPose skeleton connections
    skeleton = [
        (0, 1),   # head -> neck
        (1, 2),   # neck -> left_shoulder
        (2, 3),   # left_shoulder -> left_elbow
        (3, 4),   # left_elbow -> left_wrist
        (1, 5),   # neck -> right_shoulder
        (5, 6),   # right_shoulder -> right_elbow
        (6, 7),   # right_elbow -> right_wrist
        (1, 8),   # neck -> left_hip (through torso)
        (8, 9),   # left_hip -> left_knee
        (9, 10),  # left_knee -> left_ankle
        (10, 14), # left_ankle -> left_foot
        (1, 11),  # neck -> right_hip (through torso)
        (11, 12), # right_hip -> right_knee
        (12, 13), # right_knee -> right_ankle
        (13, 15), # right_ankle -> right_foot
        (8, 11),  # left_hip -> right_hip
    ]

    pose = pose_3d[0]  # (16, 3)

    fig = plt.figure(figsize=(16, 6))

    # 1. Original image with heatmap overlay
    ax1 = fig.add_subplot(131)
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]

    # Sum heatmaps
    heatmap_sum = heatmaps[0].sum(axis=0)
    heatmap_sum = cv2.resize(heatmap_sum, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-8)

    ax1.imshow(img_rgb)
    ax1.imshow(heatmap_norm, alpha=0.5, cmap='jet')
    ax1.set_title('Input + Heatmap')
    ax1.axis('off')

    # 2. 3D pose - front view
    ax2 = fig.add_subplot(132, projection='3d')

    xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
    ax2.scatter(xs, ys, zs, c='red', s=50)

    for i, j in skeleton:
        ax2.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b-', linewidth=2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Pose (Front View)')
    ax2.view_init(elev=0, azim=90)

    # 3. 3D pose - side view
    ax3 = fig.add_subplot(133, projection='3d')

    ax3.scatter(xs, ys, zs, c='red', s=50)

    for i, j in skeleton:
        ax3.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b-', linewidth=2)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Pose (Side View)')
    ax3.view_init(elev=0, azim=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_path}")

    plt.show()


def print_pose_3d(pose_3d):
    """Print 3D pose coordinates."""
    joint_names = [
        'head', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist',
        'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle',
        'left_foot', 'right_foot'
    ]

    pose = pose_3d[0]  # (16, 3)

    print("\n3D Pose Coordinates:")
    print("-" * 50)
    print(f"{'Joint':<15} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 50)

    for i, name in enumerate(joint_names):
        x, y, z = pose[i]
        print(f"{name:<15} {x:>10.4f} {y:>10.4f} {z:>10.4f}")

    print("-" * 50)

    # Statistics
    print(f"\nStatistics:")
    print(f"  X range: [{pose[:, 0].min():.4f}, {pose[:, 0].max():.4f}]")
    print(f"  Y range: [{pose[:, 1].min():.4f}, {pose[:, 1].max():.4f}]")
    print(f"  Z range: [{pose[:, 2].min():.4f}, {pose[:, 2].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='EgoPose FC Model Inference')
    parser.add_argument('--onnx', type=str, default='deploy_models/egopose_fc/end2end.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output visualization path')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization')

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.onnx):
        print(f"ONNX model not found: {args.onnx}")
        return

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    print(f"ONNX Model: {args.onnx}")
    print(f"Image: {args.image}")

    # Preprocess
    print("\nPreprocessing image...")
    input_tensor, original_image = preprocess_image(args.image)
    print(f"  Input shape: {input_tensor.shape}")

    # Inference
    print("\nRunning inference...")
    heatmaps, pose_3d, latency_ms = run_inference(args.onnx, input_tensor)

    print(f"\nResults:")
    print(f"  Heatmaps shape: {heatmaps.shape}")
    print(f"  Pose 3D shape: {pose_3d.shape}")
    print(f"  Latency: {latency_ms:.2f} ms ({1000/latency_ms:.1f} FPS)")

    # Print pose
    print_pose_3d(pose_3d)

    # Visualize
    if not args.no_viz:
        output_path = args.output or 'output_inference/egopose_fc_result.png'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        visualize_results(original_image, heatmaps, pose_3d, output_path)


if __name__ == '__main__':
    main()
