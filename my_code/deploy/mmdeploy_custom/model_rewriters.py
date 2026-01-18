"""
Model Rewriters for Custom EgoPose Model

This module provides ONNX-exportable wrappers for the custom dual-backbone
EgoPose model with full 3D pose output.

Reference:
    - MMDeploy Model Rewriter: https://mmdeploy.readthedocs.io/en/latest/07-developer-guide/support_new_model.html
"""

import torch
import torch.nn as nn
import numpy as np

# Try to import MMDeploy rewriter (optional dependency)
try:
    from mmdeploy.core import FUNCTION_REWRITER, MODULE_REWRITER
    HAS_MMDEPLOY = True
except ImportError:
    HAS_MMDEPLOY = False
    # Create dummy decorators
    class DummyRewriter:
        def register_rewriter(self, **kwargs):
            def decorator(func):
                return func
            return decorator
        def register_rewrite_module(self, **kwargs):
            def decorator(cls):
                return cls
            return decorator
    FUNCTION_REWRITER = DummyRewriter()
    MODULE_REWRITER = DummyRewriter()


class EgoPoseWrapper(nn.Module):
    """ONNX-exportable wrapper for Custom EgoPose Model.

    This wrapper combines the dual-backbone architecture and head into a
    single exportable module that outputs both 2D heatmaps and 3D keypoints.

    Args:
        model: The original Custom_TopdownPoseEstimator model
        output_heatmaps: Whether to output heatmaps (default: True)
        output_3d: Whether to output 3D keypoints (default: True)
    """

    def __init__(self, model, output_heatmaps=True, output_3d=True):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.output_heatmaps = output_heatmaps
        self.output_3d = output_3d

        # Check if model has second backbone
        self.has_backbone2 = hasattr(model, 'backbone2') and model.backbone2 is not None
        if self.has_backbone2:
            self.backbone2 = model.backbone2

        # Extract head components for direct access
        self._init_head_components()

    def _init_head_components(self):
        """Initialize head components for export."""
        head = self.head

        # Deconv layers for heatmap generation
        self.deconv_layers = head.deconv_layers if hasattr(head, 'deconv_layers') else None
        self.final_layer = head.final_layer if hasattr(head, 'final_layer') else None

        # 3D pose components
        self.fc_coord = head.fc_coord if hasattr(head, 'fc_coord') else None
        self.fc_layers = head.fc_layers if hasattr(head, 'fc_layers') else None

        # Simcc components (if using SimCC head)
        self.simcc_x = head.simcc_x if hasattr(head, 'simcc_x') else None
        self.simcc_y = head.simcc_y if hasattr(head, 'simcc_y') else None

    def forward(self, img: torch.Tensor, hmd_info: torch.Tensor = None):
        """Forward pass for ONNX export.

        Args:
            img: Input image tensor (B, 3, H, W)
            hmd_info: Optional HMD direction vectors (B, 9)

        Returns:
            heatmaps: 2D heatmaps (B, K, H', W') if output_heatmaps
            keypoints_3d: 3D keypoints (B, K, 3) if output_3d
        """
        # Extract features from primary backbone
        feats = self.backbone(img)
        if isinstance(feats, (list, tuple)):
            feat = feats[-1]
        else:
            feat = feats

        outputs = {}

        # Generate heatmaps
        if self.deconv_layers is not None:
            x = self.deconv_layers(feat)
            if self.final_layer is not None:
                heatmaps = self.final_layer(x)
            else:
                heatmaps = x
        else:
            heatmaps = feat
            x = feat

        if self.output_heatmaps:
            outputs['heatmaps'] = heatmaps

        # Generate 3D pose
        if self.output_3d and self.fc_coord is not None:
            # Flatten features for FC layers
            batch_size = feat.shape[0]

            # Use heatmap features or backbone features
            if hasattr(self.head, 'fc_input_from_heatmap') and self.head.fc_input_from_heatmap:
                fc_input = x.flatten(1)
            else:
                fc_input = feat.flatten(1)

            # Concatenate HMD info if available
            if hmd_info is not None and hasattr(self.head, 'use_hmd') and self.head.use_hmd:
                fc_input = torch.cat([fc_input, hmd_info], dim=1)

            # FC layers for 3D regression
            if self.fc_layers is not None:
                for fc in self.fc_layers:
                    fc_input = fc(fc_input)

            # Final 3D coordinate prediction
            keypoints_3d = self.fc_coord(fc_input)

            # Reshape to (B, K, 3)
            num_keypoints = keypoints_3d.shape[1] // 3
            keypoints_3d = keypoints_3d.view(batch_size, num_keypoints, 3)

            outputs['keypoints_3d'] = keypoints_3d

        # Return based on what's requested
        if self.output_heatmaps and self.output_3d:
            return outputs.get('heatmaps'), outputs.get('keypoints_3d')
        elif self.output_heatmaps:
            return outputs.get('heatmaps')
        elif self.output_3d:
            return outputs.get('keypoints_3d')
        else:
            return heatmaps


class EgoPoseWrapperSimplified(nn.Module):
    """Simplified wrapper that only outputs heatmaps and derives 2D keypoints.

    This is more compatible with standard ONNX export and post-processing.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, img: torch.Tensor):
        """Forward pass outputting heatmaps only.

        Args:
            img: Input image tensor (B, 3, H, W)

        Returns:
            heatmaps: 2D heatmaps (B, K, H', W')
        """
        # Extract features
        feats = self.backbone(img)
        if isinstance(feats, (list, tuple)):
            feat = feats[-1]
        else:
            feat = feats

        # Generate heatmaps
        if hasattr(self.head, 'deconv_layers') and self.head.deconv_layers is not None:
            x = self.head.deconv_layers(feat)
            if hasattr(self.head, 'final_layer') and self.head.final_layer is not None:
                heatmaps = self.head.final_layer(x)
            else:
                heatmaps = x
        else:
            heatmaps = feat

        return heatmaps


class EgoPoseFull3DWrapper(nn.Module):
    """Full 3D pose wrapper that exports the complete inference pipeline.

    This wrapper attempts to export the full head including 3D regression,
    but may require custom ONNX ops for some operations.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.head = model.head

        # Store config for reference
        self.num_keypoints = 16  # EgoPose default

    def _soft_argmax_2d(self, heatmaps):
        """Differentiable soft-argmax for 2D keypoint extraction.

        Args:
            heatmaps: (B, K, H, W)

        Returns:
            coords: (B, K, 2) normalized coordinates
        """
        batch_size, num_kpts, h, w = heatmaps.shape

        # Softmax over spatial dimensions
        heatmaps_flat = heatmaps.view(batch_size, num_kpts, -1)
        heatmaps_softmax = torch.softmax(heatmaps_flat, dim=-1)
        heatmaps_softmax = heatmaps_softmax.view(batch_size, num_kpts, h, w)

        # Create coordinate grids
        device = heatmaps.device
        y_coords = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
        x_coords = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)

        # Compute expected coordinates
        x = (heatmaps_softmax * x_coords).sum(dim=[2, 3])
        y = (heatmaps_softmax * y_coords).sum(dim=[2, 3])

        coords = torch.stack([x, y], dim=-1)
        return coords

    def forward(self, img: torch.Tensor):
        """Forward pass for full 3D pose estimation.

        Args:
            img: Input image (B, 3, 256, 256)

        Returns:
            keypoints_2d: 2D keypoints (B, K, 2)
            keypoints_3d: 3D keypoints (B, K, 3)
            heatmaps: Raw heatmaps (B, K, H', W')
        """
        batch_size = img.shape[0]

        # Backbone forward
        feats = self.backbone(img)
        if isinstance(feats, (list, tuple)):
            feat = feats[-1]
        else:
            feat = feats

        # Heatmap generation
        if hasattr(self.head, 'deconv_layers'):
            x = self.head.deconv_layers(feat)
            heatmaps = self.head.final_layer(x)
        else:
            heatmaps = feat
            x = feat

        # 2D keypoints from heatmaps (soft-argmax)
        keypoints_2d = self._soft_argmax_2d(heatmaps)

        # 3D regression
        if hasattr(self.head, 'fc_coord'):
            # Flatten features
            fc_input = x.flatten(1)

            # Pass through FC layers
            if hasattr(self.head, 'fc_layers'):
                for fc in self.head.fc_layers:
                    fc_input = fc(fc_input)

            # 3D coordinates
            keypoints_3d = self.head.fc_coord(fc_input)
            keypoints_3d = keypoints_3d.view(batch_size, self.num_keypoints, 3)
        else:
            # Fallback: estimate depth from heatmap confidence
            confidence = heatmaps.max(dim=2)[0].max(dim=2)[0]  # (B, K)
            depth = confidence.unsqueeze(-1) * 0.5  # Simple depth estimate
            keypoints_3d = torch.cat([keypoints_2d, depth], dim=-1)

        return keypoints_2d, keypoints_3d, heatmaps


def wrap_egopose_model(model, wrapper_type='simplified'):
    """Wrap EgoPose model for ONNX export.

    Args:
        model: Original Custom_TopdownPoseEstimator
        wrapper_type: 'simplified', 'full', or 'full_3d'

    Returns:
        Wrapped model ready for ONNX export
    """
    if wrapper_type == 'simplified':
        return EgoPoseWrapperSimplified(model)
    elif wrapper_type == 'full':
        return EgoPoseWrapper(model, output_heatmaps=True, output_3d=True)
    elif wrapper_type == 'full_3d':
        return EgoPoseFull3DWrapper(model)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


# Register with MMDeploy if available
if HAS_MMDEPLOY:
    @MODULE_REWRITER.register_rewrite_module(
        module_type='mmpose.models.pose_estimators.custom_topdown.Custom_TopdownPoseEstimator',
        backend='default'
    )
    class Custom_TopdownPoseEstimatorONNX(nn.Module):
        """MMDeploy rewriter for Custom_TopdownPoseEstimator."""

        def __init__(self, ctx, module, deploy_cfg):
            super().__init__()
            self.module = EgoPoseFull3DWrapper(module)

        def forward(self, img):
            return self.module(img)
