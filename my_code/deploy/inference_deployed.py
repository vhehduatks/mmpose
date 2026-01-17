"""
Run inference using deployed EgoPose model (ONNX/TensorRT).

This script loads the converted model and runs inference on images.

Usage:
    # ONNX Runtime inference
    python my_code/deploy/inference_deployed.py \
        --model-dir deploy_models/egopose_onnx \
        --image path/to/image.jpg \
        --output-dir output_deploy

    # TensorRT inference
    python my_code/deploy/inference_deployed.py \
        --model-dir deploy_models/egopose_trt \
        --backend tensorrt \
        --image path/to/image.jpg

Reference:
    - MMDeploy: https://mmdeploy.readthedocs.io/en/latest/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add mmpose to path for visualization utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_onnx_model(model_path):
    """Load ONNX model using ONNX Runtime.

    Args:
        model_path: Path to ONNX model file

    Returns:
        ONNX Runtime InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed.")
        print("Install with: pip install onnxruntime-gpu  # or onnxruntime for CPU")
        sys.exit(1)

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Try GPU first, fall back to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # Print execution provider info
    print(f"Execution providers: {session.get_providers()}")

    return session


def load_tensorrt_model(engine_path):
    """Load TensorRT engine.

    Args:
        engine_path: Path to TensorRT engine file

    Returns:
        TensorRT execution context
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("Error: TensorRT or PyCUDA not installed.")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"Loading TensorRT engine: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    return engine, context


def preprocess_image(image_path, input_size=(256, 256)):
    """Preprocess image for inference.

    Args:
        image_path: Path to input image
        input_size: Target size (H, W)

    Returns:
        Preprocessed image tensor (1, 3, H, W)
        Original image for visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    orig_img = img.copy()

    # Resize with center crop (matching training preprocessing)
    h, w = img.shape[:2]

    # Center crop to 1000x800 first (matching training bbox)
    if w == 1280 and h == 800:
        crop_x1, crop_y1 = 140, 0
        crop_x2, crop_y2 = 1140, 800
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to input size
    img = cv2.resize(img, (input_size[1], input_size[0]))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize (ImageNet mean/std)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = (img.astype(np.float32) - mean) / std

    # NHWC to NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32), orig_img


def run_onnx_inference(session, input_tensor):
    """Run inference using ONNX Runtime.

    Args:
        session: ONNX Runtime session
        input_tensor: Input tensor (1, 3, H, W)

    Returns:
        Model outputs (heatmaps)
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_tensor})

    return outputs


def run_tensorrt_inference(engine, context, input_tensor):
    """Run inference using TensorRT.

    Args:
        engine: TensorRT engine
        context: Execution context
        input_tensor: Input tensor (1, 3, H, W)

    Returns:
        Model outputs
    """
    import pycuda.driver as cuda

    # Allocate buffers
    bindings = []
    outputs = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        if engine.binding_is_input(binding):
            # Input buffer
            host_mem = input_tensor.ravel()
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            cuda.memcpy_htod(device_mem, host_mem)
            bindings.append(int(device_mem))
        else:
            # Output buffer
            host_mem = np.empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            outputs.append((host_mem, device_mem))

    # Run inference
    context.execute_v2(bindings)

    # Copy outputs back
    results = []
    for host_mem, device_mem in outputs:
        cuda.memcpy_dtoh(host_mem, device_mem)
        results.append(host_mem)

    return results


def decode_heatmaps(heatmaps, input_size=(256, 256)):
    """Decode heatmaps to keypoint coordinates.

    Args:
        heatmaps: Predicted heatmaps (1, K, H, W)
        input_size: Input image size

    Returns:
        keypoints: Decoded keypoints (K, 2)
        scores: Confidence scores (K,)
    """
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]  # Remove batch dimension

    num_keypoints = heatmaps.shape[0]
    heatmap_h, heatmap_w = heatmaps.shape[1:3]

    keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
    scores = np.zeros(num_keypoints, dtype=np.float32)

    for k in range(num_keypoints):
        hm = heatmaps[k]
        # Find maximum location
        flat_idx = np.argmax(hm)
        y, x = np.unravel_index(flat_idx, hm.shape)

        # Get confidence score
        scores[k] = hm[y, x]

        # Scale to input size
        keypoints[k, 0] = x * input_size[1] / heatmap_w
        keypoints[k, 1] = y * input_size[0] / heatmap_h

    return keypoints, scores


def visualize_results(image, keypoints, scores, output_path, threshold=0.3):
    """Visualize keypoints on image.

    Args:
        image: Original image (BGR)
        keypoints: Keypoint coordinates (K, 2)
        scores: Confidence scores (K,)
        output_path: Output image path
        threshold: Score threshold for visualization
    """
    # EgoPose skeleton connections
    skeleton = [
        [0, 1], [0, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
        [0, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15]
    ]

    # Colors
    kpt_color = (0, 255, 0)
    link_color = (255, 128, 0)

    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    # Scale keypoints to original image size
    scale_x = w / 256.0
    scale_y = h / 256.0

    scaled_kpts = keypoints.copy()
    scaled_kpts[:, 0] *= scale_x
    scaled_kpts[:, 1] *= scale_y

    # Draw skeleton
    for i, j in skeleton:
        if i < len(keypoints) and j < len(keypoints):
            if scores[i] > threshold and scores[j] > threshold:
                pt1 = tuple(scaled_kpts[i].astype(int))
                pt2 = tuple(scaled_kpts[j].astype(int))
                cv2.line(vis_img, pt1, pt2, link_color, 2)

    # Draw keypoints
    for k, (kpt, score) in enumerate(zip(scaled_kpts, scores)):
        if score > threshold:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(vis_img, (x, y), 4, kpt_color, -1)
            cv2.circle(vis_img, (x, y), 5, (255, 255, 255), 1)

    # Save result
    cv2.imwrite(output_path, vis_img)
    print(f"Visualization saved to: {output_path}")


def benchmark(inference_fn, input_tensor, num_warmup=10, num_runs=100):
    """Benchmark inference speed.

    Args:
        inference_fn: Inference function
        input_tensor: Input tensor
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Average latency in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        inference_fn(input_tensor)

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        inference_fn(input_tensor)
    elapsed = time.time() - start

    avg_latency = (elapsed / num_runs) * 1000  # ms
    fps = num_runs / elapsed

    print(f"\n=== Benchmark Results ===")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")

    return avg_latency


def main():
    parser = argparse.ArgumentParser(description='Run inference with deployed EgoPose model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing deployed model')
    parser.add_argument('--backend', type=str, default='onnxruntime',
                        choices=['onnxruntime', 'tensorrt'],
                        help='Inference backend')
    parser.add_argument('--image', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output-dir', type=str, default='output_deploy',
                        help='Output directory')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load deploy config
    config_path = os.path.join(args.model_dir, 'deploy.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            deploy_config = json.load(f)
        input_size = tuple(deploy_config.get('input_shape', [1, 3, 256, 256])[2:])
    else:
        input_size = (256, 256)

    # Load model
    if args.backend == 'onnxruntime':
        model_path = os.path.join(args.model_dir, 'end2end.onnx')
        session = load_onnx_model(model_path)
        inference_fn = lambda x: run_onnx_inference(session, x)
    elif args.backend == 'tensorrt':
        engine_path = os.path.join(args.model_dir, 'end2end.engine')
        engine, context = load_tensorrt_model(engine_path)
        inference_fn = lambda x: run_tensorrt_inference(engine, context, x)
    else:
        print(f"Unsupported backend: {args.backend}")
        sys.exit(1)

    # Preprocess image
    print(f"\nProcessing image: {args.image}")
    input_tensor, orig_img = preprocess_image(args.image, input_size)
    print(f"Input shape: {input_tensor.shape}")

    # Run inference
    print("Running inference...")
    start = time.time()
    outputs = inference_fn(input_tensor)
    elapsed = (time.time() - start) * 1000
    print(f"Inference time: {elapsed:.2f} ms")

    # Decode results
    heatmaps = outputs[0]
    print(f"Output heatmaps shape: {heatmaps.shape}")

    keypoints, scores = decode_heatmaps(heatmaps, input_size)
    print(f"Detected {len(keypoints)} keypoints")

    # Visualize
    output_path = os.path.join(args.output_dir, 'result.jpg')
    visualize_results(orig_img, keypoints, scores, output_path)

    # Run benchmark if requested
    if args.benchmark:
        benchmark(inference_fn, input_tensor)

    print("\nInference completed!")


if __name__ == '__main__':
    main()
