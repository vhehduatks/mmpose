"""
Convert EgoPose Model with Full FC Layers for 3D Pose Regression

This script properly exports the encoder + pose_decoder FC layers
for accurate 3D pose estimation.

Architecture:
    Backbone → Deconv → Heatmap (16, 32, 32)
                           ↓
                       Encoder ← HMD (9)
                           ↓
                       Latent (64)
                           ↓
                     pose_decoder (FC layers)
                           ↓
                       3D Pose (16, 3)

Usage:
    python my_code/deploy/convert_egopose_fc.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache/best_*.pth \
        --output-dir deploy_models/egopose_fc
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config
from mmpose.apis import init_model


class EgoPoseFullFCWrapper(nn.Module):
    """Full EgoPose wrapper with proper FC layers for 3D regression.

    This wrapper includes:
    - Backbone (ResNet)
    - Deconv layers (heatmap generation)
    - Encoder (heatmap + HMD → latent)
    - pose_decoder (latent → 3D pose)
    """

    def __init__(self, model, use_hmd=False):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.use_hmd = use_hmd

        # Extract head components
        self.deconv_layers = self.head.deconv_layers
        self.final_layer = self.head.final_layer
        self.encoder = self.head.encoder
        self.pose_decoder = self.head.pose_decoder

        # HMD processing
        if use_hmd and hasattr(self.head, 'hmd_linear'):
            self.hmd_linear = self.head.hmd_linear

    def forward(self, image: torch.Tensor, hmd: torch.Tensor = None):
        """Forward pass with full FC pipeline.

        Args:
            image: (B, 3, 256, 256) input image
            hmd: (B, 9) HMD direction vectors (optional)

        Returns:
            heatmaps: (B, 16, 32, 32)
            pose_3d: (B, 16, 3)
        """
        batch_size = image.shape[0]

        # Backbone
        feats = self.backbone(image)
        if isinstance(feats, (list, tuple)):
            feat = feats[-1]
        else:
            feat = feats

        # Deconv → Heatmap
        x = self.deconv_layers(feat)
        heatmaps = self.final_layer(x)  # (B, 16, 32, 32)

        # Encoder: heatmap + HMD → latent
        if self.use_hmd and hmd is not None:
            latent = self.encoder(heatmaps, hmd)  # (B, 64)
        else:
            # Create dummy HMD if not provided
            dummy_hmd = torch.zeros(batch_size, 9, device=image.device)
            latent = self.encoder(heatmaps, dummy_hmd)

        # pose_decoder: latent → 3D pose
        pose_3d_flat = self.pose_decoder(latent)  # (B, 48)
        pose_3d = pose_3d_flat.view(batch_size, 16, 3)  # (B, 16, 3)

        return heatmaps, pose_3d


class EgoPoseNoHMDWrapper(nn.Module):
    """EgoPose wrapper without HMD input (simpler export).

    Uses zero HMD vectors for inference.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

        self.deconv_layers = self.head.deconv_layers
        self.final_layer = self.head.final_layer
        self.encoder = self.head.encoder
        self.pose_decoder = self.head.pose_decoder

    def forward(self, image: torch.Tensor):
        """Forward pass without HMD input.

        Args:
            image: (B, 3, 256, 256)

        Returns:
            heatmaps: (B, 16, 32, 32)
            pose_3d: (B, 16, 3)
        """
        batch_size = image.shape[0]
        device = image.device

        # Backbone → Heatmap
        feats = self.backbone(image)
        feat = feats[-1] if isinstance(feats, (list, tuple)) else feats

        x = self.deconv_layers(feat)
        heatmaps = self.final_layer(x)

        # Encoder with zero HMD
        dummy_hmd = torch.zeros(batch_size, 9, device=device)
        latent = self.encoder(heatmaps, dummy_hmd)

        # pose_decoder → 3D
        pose_3d = self.pose_decoder(latent).view(batch_size, 16, 3)

        return heatmaps, pose_3d


def export_model(config_path, checkpoint_path, output_dir, use_hmd=False):
    """Export EgoPose model with full FC layers."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Create wrapper
    if use_hmd:
        wrapper = EgoPoseFullFCWrapper(model, use_hmd=True)
        input_names = ['image', 'hmd']
        dummy_inputs = (
            torch.randn(1, 3, 256, 256).cuda(),
            torch.randn(1, 9).cuda()
        )
        dynamic_axes = {
            'image': {0: 'batch'},
            'hmd': {0: 'batch'},
            'heatmaps': {0: 'batch'},
            'pose_3d': {0: 'batch'}
        }
    else:
        wrapper = EgoPoseNoHMDWrapper(model)
        input_names = ['image']
        dummy_inputs = torch.randn(1, 3, 256, 256).cuda()
        dynamic_axes = {
            'image': {0: 'batch'},
            'heatmaps': {0: 'batch'},
            'pose_3d': {0: 'batch'}
        }

    wrapper.eval()
    wrapper.cuda()

    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        if use_hmd:
            heatmaps, pose_3d = wrapper(*dummy_inputs)
        else:
            heatmaps, pose_3d = wrapper(dummy_inputs)

    print(f"  Heatmaps: {heatmaps.shape}")
    print(f"  Pose 3D: {pose_3d.shape}")
    print(f"  Pose 3D sample:\n{pose_3d[0, :3]}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'end2end.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=['heatmaps', 'pose_3d'],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )

    print("ONNX export successful!")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("\nONNX verification passed!")

        print("\nInputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {dims}")

        print("Outputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {dims}")

        # Count operations
        op_types = {}
        for node in onnx_model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

        print("\nONNX Operations:")
        for op, count in sorted(op_types.items()):
            print(f"  {op}: {count}")

    except Exception as e:
        print(f"Verification warning: {e}")

    # Save config
    deploy_config = {
        'backend': 'onnxruntime',
        'use_hmd': use_hmd,
        'input_size': [256, 256],
        'num_keypoints': 16,
        'heatmap_size': [32, 32],
    }

    config_file = os.path.join(output_dir, 'deploy.json')
    with open(config_file, 'w') as f:
        json.dump(deploy_config, f, indent=2)

    print(f"\nConfig saved: {config_file}")
    print(f"\n=== Export Complete ===")
    print(f"Output: {output_dir}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='deploy_models/egopose_fc')
    parser.add_argument('--use-hmd', action='store_true', help='Include HMD input')

    args = parser.parse_args()

    # Resolve wildcard
    checkpoint = args.checkpoint
    if '*' in checkpoint:
        matches = glob.glob(checkpoint)
        if matches:
            checkpoint = sorted(matches)[-1]
        else:
            print(f"No checkpoint found: {args.checkpoint}")
            return

    export_model(args.config, checkpoint, args.output_dir, args.use_hmd)


if __name__ == '__main__':
    main()
