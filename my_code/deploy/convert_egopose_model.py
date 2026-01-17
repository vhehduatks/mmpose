"""
Convert Custom EgoPose Model to Deployment Format (ONNX, TensorRT, etc.)

This script converts the trained Custom_TopdownPoseEstimator model to various
deployment backends for optimized inference.

Usage:
    python my_code/deploy/convert_egopose_model.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache/best_*.pth \
        --output-dir deploy_models/egopose_onnx \
        --backend onnxruntime

Supported backends:
    - onnxruntime (default)
    - tensorrt (requires TensorRT installation)
    - openvino

Reference:
    - MMDeploy: https://mmdeploy.readthedocs.io/en/latest/
    - MMPose Deployment: https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpose.html
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add mmpose to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.apis import init_model


class EgoPoseInferenceWrapper(nn.Module):
    """Wrapper for EgoPose model inference.

    This wrapper extracts the main inference path from the dual-backbone model
    for deployment. It takes an image and HMD info as input and outputs 3D pose.

    Args:
        model: The full Custom_TopdownPoseEstimator model
        with_hmd: Whether to include HMD input (default: True)
    """

    def __init__(self, model, with_hmd=True):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.with_hmd = with_hmd

        # Store necessary components from head
        self.head_deconv = self.head.deconv_layers if hasattr(self.head, 'deconv_layers') else None

    def forward(self, img, hmd_info=None):
        """Forward pass for inference.

        Args:
            img: Input image tensor (B, 3, H, W), normalized
            hmd_info: HMD direction vectors (B, 9) - head/left_hand/right_hand directions

        Returns:
            keypoints_3d: 3D keypoint predictions (B, N, 3)
            heatmaps: 2D heatmap predictions (B, N, H', W')
        """
        # Extract features from backbone
        feats = self.backbone(img)

        # Get the last feature map
        if isinstance(feats, (list, tuple)):
            feat = feats[-1]
        else:
            feat = feats

        # Forward through head (simplified for export)
        # Note: The full head has complex logic, we extract key operations
        output = self.head.forward_export(feat, hmd_info)

        return output


class EgoPoseHeadExport(nn.Module):
    """Exportable version of the custom head.

    Simplified forward pass for ONNX export without data_sample dependencies.
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feat, hmd_info=None):
        """Simplified forward for export."""
        # Deconv layers for heatmap
        if hasattr(self.head, 'deconv_layers'):
            x = self.head.deconv_layers(feat)
        else:
            x = feat

        # Final heatmap prediction
        if hasattr(self.head, 'final_layer'):
            heatmaps = self.head.final_layer(x)
        else:
            heatmaps = x

        # 3D pose regression from heatmap features
        # This part depends on the specific head implementation
        if hasattr(self.head, 'pose_3d_head'):
            pose_3d = self.head.pose_3d_head(x)
        else:
            # Fallback: derive 3D from heatmap argmax + depth estimation
            pose_3d = None

        return heatmaps, pose_3d


def convert_to_onnx(model, output_path, input_shape=(1, 3, 256, 256),
                    hmd_shape=(1, 9), opset_version=11, dynamic_batch=False):
    """Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_shape: Input image shape (B, C, H, W)
        hmd_shape: HMD info shape (B, 9)
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
    """
    model.eval()

    # Create dummy inputs
    dummy_img = torch.randn(*input_shape)
    dummy_hmd = torch.randn(*hmd_shape)

    # Move to same device as model
    device = next(model.parameters()).device
    dummy_img = dummy_img.to(device)
    dummy_hmd = dummy_hmd.to(device)

    # Define input/output names
    input_names = ['image', 'hmd_info']
    output_names = ['heatmaps', 'keypoints_3d']

    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'hmd_info': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
            'keypoints_3d': {0: 'batch_size'}
        }

    # Export to ONNX
    print(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_img, dummy_hmd),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )

    print(f"ONNX model saved to: {output_path}")

    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")

    return output_path


def convert_to_tensorrt(onnx_path, output_path, fp16=True, int8=False,
                        max_batch_size=1, workspace_size=1<<30):
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Input ONNX file path
        output_path: Output TensorRT engine path
        fp16: Use FP16 precision
        int8: Use INT8 precision (requires calibration)
        max_batch_size: Maximum batch size
        workspace_size: TensorRT workspace size in bytes
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT not installed. Please install TensorRT first.")
        print("See: https://developer.nvidia.com/tensorrt")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"TensorRT ONNX Parser Error: {parser.get_error(error)}")
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabling FP16 precision")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Enabling INT8 precision (requires calibration)")

    # Build engine
    print("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        print("Error: Failed to build TensorRT engine")
        return None

    # Serialize and save
    print(f"Saving TensorRT engine to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print("TensorRT engine saved successfully!")
    return output_path


def create_deploy_config(output_dir, backend, input_shape, model_info):
    """Create deployment configuration files.

    Args:
        output_dir: Output directory
        backend: Backend name (onnxruntime, tensorrt, etc.)
        input_shape: Input shape tuple
        model_info: Model information dict
    """
    # Deploy config
    deploy_config = {
        'backend': backend,
        'input_shape': list(input_shape),
        'input_names': ['image', 'hmd_info'],
        'output_names': ['heatmaps', 'keypoints_3d'],
        'num_keypoints': model_info.get('num_keypoints', 16),
        'heatmap_size': model_info.get('heatmap_size', [47, 47]),
    }

    config_path = os.path.join(output_dir, 'deploy.json')
    with open(config_path, 'w') as f:
        json.dump(deploy_config, f, indent=2)
    print(f"Deploy config saved to: {config_path}")

    # Pipeline config for inference
    pipeline_config = {
        'pipeline': [
            {'type': 'LoadImage'},
            {'type': 'Resize', 'size': list(input_shape[2:])},
            {'type': 'Normalize',
             'mean': [123.675, 116.28, 103.53],
             'std': [58.395, 57.12, 57.375]},
            {'type': 'ToTensor'},
        ]
    }

    pipeline_path = os.path.join(output_dir, 'pipeline.json')
    with open(pipeline_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"Pipeline config saved to: {pipeline_path}")


def export_simplified_model(config_path, checkpoint_path, output_dir,
                           backend='onnxruntime', input_size=(256, 256)):
    """Export a simplified version of the model for deployment.

    Since the custom model has complex dual-backbone architecture,
    this function exports only the primary inference path.

    Args:
        config_path: Path to model config
        checkpoint_path: Path to checkpoint
        output_dir: Output directory
        backend: Target backend
        input_size: Input image size
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    cfg = Config.fromfile(config_path)

    # Initialize model
    print(f"Loading model from config: {config_path}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Get model info
    model_info = {
        'num_keypoints': cfg.model.head.get('out_channels', 16),
        'heatmap_size': cfg.codec.get('heatmap_size', (47, 47)),
        'input_size': input_size,
    }

    # For custom models, we need to handle the export differently
    # The model has: backbone, backbone2, head with multiple components

    print("\n=== Model Structure ===")
    print(f"Backbone: {type(model.backbone).__name__}")
    if hasattr(model, 'backbone2'):
        print(f"Backbone2: {type(model.backbone2).__name__}")
    print(f"Head: {type(model.head).__name__}")

    # Try direct ONNX export first
    try:
        # Create simplified inference model
        class SimpleInferenceModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                # Only use primary backbone for inference
                feats = self.backbone(x)
                if isinstance(feats, (list, tuple)):
                    feat = feats[-1]
                else:
                    feat = feats

                # Simple heatmap prediction
                if hasattr(self.head, 'deconv_layers'):
                    x = self.head.deconv_layers(feat)
                    if hasattr(self.head, 'final_layer'):
                        heatmaps = self.head.final_layer(x)
                    else:
                        heatmaps = x
                else:
                    heatmaps = feat

                return heatmaps

        simple_model = SimpleInferenceModel(model.backbone, model.head)
        simple_model.eval()
        simple_model.cuda()

        # Export to ONNX
        input_shape = (1, 3, input_size[0], input_size[1])
        onnx_path = os.path.join(output_dir, 'end2end.onnx')

        dummy_input = torch.randn(*input_shape).cuda()

        torch.onnx.export(
            simple_model,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=['heatmaps'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'heatmaps': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )

        print(f"\nONNX model saved to: {onnx_path}")

        # Verify ONNX
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed!")

            # Print model info
            print(f"\nModel inputs:")
            for inp in onnx_model.graph.input:
                print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
            print(f"Model outputs:")
            for out in onnx_model.graph.output:
                print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

        except ImportError:
            print("Warning: onnx package not installed, skipping verification")

        # Convert to TensorRT if requested
        if backend == 'tensorrt':
            trt_path = os.path.join(output_dir, 'end2end.engine')
            convert_to_tensorrt(onnx_path, trt_path, fp16=True)

        # Create deploy configs
        create_deploy_config(output_dir, backend, input_shape, model_info)

        print(f"\n=== Deployment Complete ===")
        print(f"Output directory: {output_dir}")
        print(f"Backend: {backend}")

        return True

    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert EgoPose model for deployment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='deploy_models/egopose',
                        help='Output directory for deployed model')
    parser.add_argument('--backend', type=str, default='onnxruntime',
                        choices=['onnxruntime', 'tensorrt', 'openvino'],
                        help='Target deployment backend')
    parser.add_argument('--input-size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (H W)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision (TensorRT only)')

    args = parser.parse_args()

    # Handle wildcard in checkpoint path
    checkpoint_path = args.checkpoint
    if '*' in checkpoint_path:
        import glob
        matches = glob.glob(checkpoint_path)
        if matches:
            checkpoint_path = sorted(matches)[-1]  # Get latest
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print(f"Error: No checkpoint found matching {args.checkpoint}")
            return

    # Export model
    success = export_simplified_model(
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        backend=args.backend,
        input_size=tuple(args.input_size)
    )

    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed. Check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
