"""
Run inference with Full 3D EgoPose ONNX model.

Usage:
    python my_code/deploy/inference_full3d.py \
        --model-dir deploy_models/egopose_full3d \
        --image path/to/image.png \
        --output-dir output_deploy_3d
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# EgoPose skeleton
EGOPOSE_SKELETON = [
    [0, 1], [0, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
    [0, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15]
]

EGOPOSE_COLORS = [
    (51, 153, 255), (51, 153, 255), (51, 153, 255), (0, 255, 0), (0, 255, 0),
    (51, 153, 255), (255, 128, 0), (255, 128, 0), (51, 153, 255), (0, 255, 0),
    (0, 255, 0), (0, 255, 0), (51, 153, 255), (255, 128, 0), (255, 128, 0), (255, 128, 0)
]


def load_onnx_model(model_path):
    """Load ONNX model."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    print(f"Loaded model: {model_path}")
    print(f"Providers: {session.get_providers()}")

    return session


def preprocess_image(image_path, input_size=(256, 256)):
    """Preprocess image for inference."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load: {image_path}")

    # Handle RGBA
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    orig_img = img.copy()
    h, w = img.shape[:2]

    # Center crop for fisheye (1280x800 -> 1000x800)
    if w == 1280 and h == 800:
        img = img[0:800, 140:1140]

    # Resize
    img = cv2.resize(img, (input_size[1], input_size[0]))

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = (img.astype(np.float32) - mean) / std

    # NHWC to NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    return img.astype(np.float32), orig_img


def run_inference(session, input_tensor):
    """Run ONNX inference."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_tensor})

    return dict(zip(output_names, outputs))


def visualize_results(image, keypoints_2d, keypoints_3d, output_path, input_size=(256, 256)):
    """Visualize 2D and 3D pose results."""
    h, w = image.shape[:2]

    # Scale 2D keypoints to image size
    kpts_2d = keypoints_2d[0].copy()  # (16, 2)
    kpts_2d[:, 0] *= w
    kpts_2d[:, 1] *= h

    # 3D keypoints
    kpts_3d = keypoints_3d[0].copy()  # (16, 3)

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # 2D pose on image
    ax1 = fig.add_subplot(1, 3, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)

    # Draw skeleton
    for i, (start, end) in enumerate(EGOPOSE_SKELETON):
        color = np.array(EGOPOSE_COLORS[i]) / 255.0
        ax1.plot([kpts_2d[start, 0], kpts_2d[end, 0]],
                 [kpts_2d[start, 1], kpts_2d[end, 1]],
                 color=color, linewidth=2)

    # Draw keypoints
    for i, kpt in enumerate(kpts_2d):
        color = np.array(EGOPOSE_COLORS[i]) / 255.0
        ax1.scatter(kpt[0], kpt[1], c=[color], s=50, zorder=5)

    ax1.set_title('2D Pose')
    ax1.axis('off')

    # 3D pose
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # Draw 3D skeleton
    for i, (start, end) in enumerate(EGOPOSE_SKELETON):
        color = np.array(EGOPOSE_COLORS[i]) / 255.0
        ax2.plot([kpts_3d[start, 0], kpts_3d[end, 0]],
                 [kpts_3d[start, 1], kpts_3d[end, 1]],
                 [kpts_3d[start, 2], kpts_3d[end, 2]],
                 color=color, linewidth=2)

    # Draw 3D keypoints
    colors = [np.array(c) / 255.0 for c in EGOPOSE_COLORS]
    ax2.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],
                c=colors, s=50, depthshade=True)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Pose')

    # Set equal aspect ratio
    max_range = np.abs(kpts_3d).max() * 1.2
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])
    ax2.view_init(elev=15, azim=70)

    # 3D pose (different angle)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    for i, (start, end) in enumerate(EGOPOSE_SKELETON):
        color = np.array(EGOPOSE_COLORS[i]) / 255.0
        ax3.plot([kpts_3d[start, 0], kpts_3d[end, 0]],
                 [kpts_3d[start, 1], kpts_3d[end, 1]],
                 [kpts_3d[start, 2], kpts_3d[end, 2]],
                 color=color, linewidth=2)

    ax3.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],
                c=colors, s=50, depthshade=True)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Pose (Side View)')
    ax3.set_xlim([-max_range, max_range])
    ax3.set_ylim([-max_range, max_range])
    ax3.set_zlim([-max_range, max_range])
    ax3.view_init(elev=0, azim=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def benchmark(session, input_tensor, num_warmup=10, num_runs=100):
    """Benchmark inference speed."""
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: input_tensor})

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: input_tensor})
    elapsed = time.time() - start

    avg_latency = (elapsed / num_runs) * 1000
    fps = num_runs / elapsed

    print(f"\n=== Benchmark Results ===")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")

    return avg_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='output_deploy_3d')
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model_path = os.path.join(args.model_dir, 'end2end.onnx')
    session = load_onnx_model(model_path)

    # Load config
    config_path = os.path.join(args.model_dir, 'deploy.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        input_size = tuple(config.get('input_size', [256, 256]))
    else:
        input_size = (256, 256)

    # Preprocess
    print(f"\nProcessing: {args.image}")
    input_tensor, orig_img = preprocess_image(args.image, input_size)
    print(f"Input shape: {input_tensor.shape}")

    # Inference
    print("Running inference...")
    start = time.time()
    outputs = run_inference(session, input_tensor)
    elapsed = (time.time() - start) * 1000
    print(f"Inference time: {elapsed:.2f} ms")

    # Print outputs
    print("\nOutputs:")
    for name, arr in outputs.items():
        print(f"  {name}: {arr.shape}")

    # Visualize
    output_path = os.path.join(args.output_dir, 'result_3d.png')
    visualize_results(
        orig_img,
        outputs.get('keypoints_2d'),
        outputs.get('keypoints_3d'),
        output_path,
        input_size
    )

    # Benchmark
    if args.benchmark:
        benchmark(session, input_tensor)

    print("\nDone!")


if __name__ == '__main__':
    main()
