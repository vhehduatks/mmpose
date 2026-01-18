"""
Convert Custom EgoPose Model with Full 3D Pose Output

This script exports the complete EgoPose model including:
- 2D heatmaps
- 2D keypoint coordinates (from soft-argmax)
- 3D keypoint coordinates

Usage:
    python my_code/deploy/convert_egopose_full3d.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache/best_*.pth \
        --output-dir deploy_models/egopose_full3d \
        --backend onnxruntime

Reference:
    - MMDeploy: https://mmdeploy.readthedocs.io/en/latest/
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config
from mmpose.apis import init_model

# Import custom wrappers
from mmdeploy_custom.model_rewriters import (
    EgoPoseWrapper,
    EgoPoseWrapperSimplified,
    EgoPoseFull3DWrapper,
    wrap_egopose_model
)


def export_full_3d_model(config_path, checkpoint_path, output_dir,
                         backend='onnxruntime', input_size=(256, 256),
                         wrapper_type='full_3d'):
    """Export EgoPose model with full 3D pose output.

    Args:
        config_path: Path to model config
        checkpoint_path: Path to checkpoint
        output_dir: Output directory
        backend: Target backend ('onnxruntime', 'tensorrt')
        input_size: Input image size (H, W)
        wrapper_type: 'simplified', 'full', or 'full_3d'
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Get model info
    num_keypoints = cfg.model.head.get('out_channels', 16)
    heatmap_size = cfg.codec.get('heatmap_size', (47, 47))

    print(f"\n=== Model Info ===")
    print(f"Backbone: {type(model.backbone).__name__}")
    print(f"Head: {type(model.head).__name__}")
    print(f"Num keypoints: {num_keypoints}")
    print(f"Heatmap size: {heatmap_size}")

    # Wrap model for export
    print(f"\nWrapping model with: {wrapper_type}")
    wrapped_model = wrap_egopose_model(model, wrapper_type=wrapper_type)
    wrapped_model.eval()
    wrapped_model.cuda()

    # Prepare dummy input
    input_shape = (1, 3, input_size[0], input_size[1])
    dummy_input = torch.randn(*input_shape).cuda()

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = wrapped_model(dummy_input)

    if isinstance(outputs, tuple):
        print(f"Output shapes:")
        for i, out in enumerate(outputs):
            if out is not None:
                print(f"  Output {i}: {out.shape}")
    else:
        print(f"Output shape: {outputs.shape}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'end2end.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")

    # Determine output names based on wrapper type
    if wrapper_type == 'full_3d':
        output_names = ['keypoints_2d', 'keypoints_3d', 'heatmaps']
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'keypoints_2d': {0: 'batch_size'},
            'keypoints_3d': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
        }
    elif wrapper_type == 'full':
        output_names = ['heatmaps', 'keypoints_3d']
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
            'keypoints_3d': {0: 'batch_size'},
        }
    else:
        output_names = ['heatmaps']
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
        }

    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        print("ONNX export successful!")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nTrying simplified export...")

        # Fallback to simplified wrapper
        wrapped_model = wrap_egopose_model(model, wrapper_type='simplified')
        wrapped_model.eval()
        wrapped_model.cuda()

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=['heatmaps'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'heatmaps': {0: 'batch_size'},
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        output_names = ['heatmaps']
        print("Simplified ONNX export successful!")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("\nONNX model verification passed!")

        print("\nModel inputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {dims}")

        print("Model outputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {dims}")

    except ImportError:
        print("Warning: onnx package not available for verification")
    except Exception as e:
        print(f"Warning: ONNX verification issue: {e}")

    # Convert to TensorRT if requested
    if backend == 'tensorrt':
        print("\nConverting to TensorRT...")
        trt_path = os.path.join(output_dir, 'end2end.engine')
        convert_to_tensorrt(onnx_path, trt_path)

    # Save deploy config
    deploy_config = {
        'backend': backend,
        'wrapper_type': wrapper_type,
        'input_shape': list(input_shape),
        'input_names': ['image'],
        'output_names': output_names,
        'num_keypoints': num_keypoints,
        'heatmap_size': list(heatmap_size),
        'input_size': list(input_size),
    }

    config_path = os.path.join(output_dir, 'deploy.json')
    with open(config_path, 'w') as f:
        json.dump(deploy_config, f, indent=2)
    print(f"\nDeploy config saved to: {config_path}")

    # Save pipeline config
    pipeline_config = {
        'preprocess': {
            'resize': list(input_size),
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True,
        },
        'postprocess': {
            'decode_heatmap': True,
            'heatmap_size': list(heatmap_size),
            'num_keypoints': num_keypoints,
        }
    }

    pipeline_path = os.path.join(output_dir, 'pipeline.json')
    with open(pipeline_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"Pipeline config saved to: {pipeline_path}")

    print(f"\n=== Export Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Backend: {backend}")
    print(f"Wrapper: {wrapper_type}")

    return True


def convert_to_tensorrt(onnx_path, output_path, fp16=True):
    """Convert ONNX to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed, skipping conversion")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"Parser error: {parser.get_error(error)}")
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)

    if engine:
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to: {output_path}")
        return output_path

    return None


def main():
    parser = argparse.ArgumentParser(description='Export EgoPose with full 3D output')
    parser.add_argument('--config', type=str, required=True,
                        help='Model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint')
    parser.add_argument('--output-dir', type=str, default='deploy_models/egopose_full3d',
                        help='Output directory')
    parser.add_argument('--backend', type=str, default='onnxruntime',
                        choices=['onnxruntime', 'tensorrt'],
                        help='Target backend')
    parser.add_argument('--wrapper', type=str, default='full_3d',
                        choices=['simplified', 'full', 'full_3d'],
                        help='Wrapper type')
    parser.add_argument('--input-size', type=int, nargs=2, default=[256, 256],
                        help='Input size (H W)')

    args = parser.parse_args()

    # Resolve checkpoint path with wildcard
    checkpoint_path = args.checkpoint
    if '*' in checkpoint_path:
        matches = glob.glob(checkpoint_path)
        if matches:
            checkpoint_path = sorted(matches)[-1]
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoint found: {args.checkpoint}")
            return

    export_full_3d_model(
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        backend=args.backend,
        input_size=tuple(args.input_size),
        wrapper_type=args.wrapper
    )


if __name__ == '__main__':
    main()
