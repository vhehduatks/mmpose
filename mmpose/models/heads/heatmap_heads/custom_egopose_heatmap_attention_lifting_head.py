# Copyright (c) OpenMMLab. All rights reserved.
"""
Per-Joint Heatmap Attention Lifting Head

Key improvement over Global Attention Lifting:
- 16 heatmap tokens instead of 1 global token
- Each joint's heatmap encoded as individual token
- K/V: [Backbone 64 + Heatmap 16] = 80 tokens

Architecture:
    Backbone feat [2048, 8, 8]
           │
           ├──────────────────────────────────────────┐
           │                                          │
           ▼ (Deconv)                                 ▼ (Conv → flatten)
    Heatmap [16, 47, 47]                    Backbone tokens [64, D]
           │                                          │
           ├─────────┐                                │
           │         │                                │
           ▼         ▼                                │
      soft_argmax   PerJointEncoder                   │
           │         │                                │
           ▼         ▼                                │
      2D coords   Heatmap Tokens [16, D] ─────────────┤
      [16, 2]                                         │
           │                                          │
           ▼                                          │
      Joint embed [16, D]                             │
           │                                          │
           └──────── Cross-Attention ─────────────────┘
                     Q: joints [16, D]
                     K/V: [backbone 64 + heatmap 16] = [80, D]
                              │
                              ▼
                     Depth-aware joints [16, D]
                              │
                              + HMD Cross-Attention
                              ▼
                        Self-Attention × 2
                              │
                              ▼
                        3D Pose [16, 3]

Expected improvements:
- Each joint's heatmap info preserved (not compressed to 1 token)
- Joint i can attend to its own heatmap token i
- Cross-joint attention: occluded joints reference visible ones
- 20% of K/V is heatmap info (vs 1.5% with global token)
"""

from typing import Optional, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData, InstanceData

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead

import numpy as np
import math
from .blocks import EfficientHeatmapDecoder

OptIntSeq = Optional[Sequence[int]]


def soft_argmax_2d(heatmaps: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Differentiable 2D coordinate extraction from heatmaps.
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    heatmaps_flat = heatmaps.view(B, K, -1)
    heatmaps_soft = F.softmax(heatmaps_flat / temperature, dim=-1)

    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    x = (heatmaps_soft * xx_flat.view(1, 1, -1)).sum(dim=-1)
    y = (heatmaps_soft * yy_flat.view(1, 1, -1)).sum(dim=-1)

    coords = torch.stack([x, y], dim=-1)
    confidence = heatmaps.view(B, K, -1).max(dim=-1)[0]

    return coords, confidence


class PerJointHeatmapEncoder(nn.Module):
    """
    Encode each joint's heatmap to individual token.

    Input: Heatmap [B, 16, 47, 47]
    Output: Tokens [B, 16, D]

    Each joint's heatmap [1, 47, 47] → token [D]
    """

    def __init__(self, num_joints: int = 16, output_dim: int = 64):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim

        # Shared encoder for all joints
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # [32, 24, 24]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [128, 6, 6]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [128, 1, 1]
        )

        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, heatmaps: Tensor) -> Tensor:
        """
        Args:
            heatmaps: [B, 16, 47, 47]

        Returns:
            tokens: [B, 16, D]
        """
        B, K, H, W = heatmaps.shape

        # Reshape to process all joints at once: [B*16, 1, 47, 47]
        heatmaps_flat = heatmaps.view(B * K, 1, H, W)

        # Encode: [B*16, 128, 1, 1]
        features = self.encoder(heatmaps_flat)
        features = features.view(B * K, -1)  # [B*16, 128]

        # Project: [B*16, D]
        tokens = self.fc(features)

        # Reshape back: [B, 16, D]
        tokens = tokens.view(B, K, -1)

        return tokens


class HeatmapBackboneCrossAttention(nn.Module):
    """
    Cross-attention with Per-Joint Heatmap Tokens.

    K/V: [Backbone tokens [B, 64, D] + Heatmap tokens [B, 16, D]] = [B, 80, D]
    Q: Joint tokens [B, 16, D]

    Each joint can attend to:
    1. Spatial backbone features (64 positions) - depth cues
    2. Per-joint heatmap tokens (16 joints) - distribution info
    """

    def __init__(
        self,
        joint_dim: int = 64,
        backbone_channels: int = 2048,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.joint_dim = joint_dim

        # Backbone feature → tokens
        self.backbone_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, joint_dim * 2, kernel_size=1),
            nn.BatchNorm2d(joint_dim * 2),
            nn.ReLU()
        )
        self.backbone_to_kv = nn.Linear(joint_dim * 2, joint_dim)

        # Heatmap tokens projection (already [B, 16, D])
        self.heatmap_proj = nn.Sequential(
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim)
        )

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(joint_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        joint_tokens: Tensor,
        backbone_feat: Tensor,
        heatmap_tokens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            joint_tokens: [B, 16, D] joint queries
            backbone_feat: [B, 2048, 8, 8] backbone features
            heatmap_tokens: [B, 16, D] per-joint heatmap tokens

        Returns:
            depth_aware_joints: [B, 16, D]
            attn_weights: [B, 16, 80] attention map (64 spatial + 16 heatmap)
        """
        B = joint_tokens.size(0)

        # Backbone → spatial tokens [B, 64, D]
        backbone_tokens = self.backbone_proj(backbone_feat)  # [B, D*2, 8, 8]
        backbone_tokens = backbone_tokens.flatten(2).transpose(1, 2)  # [B, 64, D*2]
        backbone_kv = self.backbone_to_kv(backbone_tokens)  # [B, 64, D]

        # Heatmap tokens [B, 16, D]
        heatmap_kv = self.heatmap_proj(heatmap_tokens)  # [B, 16, D]

        # Concatenate: [B, 80, D] = [backbone 64 + heatmap 16]
        combined_kv = torch.cat([backbone_kv, heatmap_kv], dim=1)

        # Cross attention
        attn_out, attn_weights = self.cross_attn(
            query=joint_tokens,  # [B, 16, D]
            key=combined_kv,     # [B, 80, D]
            value=combined_kv
        )
        # attn_weights: [B, 16, 80]
        #   - [:, :, :64] → backbone spatial attention
        #   - [:, :, 64:] → heatmap token attention

        # Residual + norm
        out = self.norm(joint_tokens + self.dropout(attn_out))

        return out, attn_weights


class HMDCrossAttention(nn.Module):
    """Cross-attention: Joint tokens query HMD tokens for 3D reference."""

    def __init__(
        self,
        joint_dim: int = 64,
        hmd_dim: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Linear(joint_dim * 2, joint_dim * 3)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(joint_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, joint_tokens: Tensor, hmd_info: Tensor) -> Tuple[Tensor, Tensor]:
        B = joint_tokens.size(0)
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        attn_out, attn_weights = self.cross_attn(
            query=joint_tokens,
            key=hmd_tokens,
            value=hmd_tokens
        )

        out = self.norm(joint_tokens + self.dropout(attn_out))
        return out, attn_weights


class JointSelfAttention(nn.Module):
    """Self-attention among joint tokens for structural relationships."""

    def __init__(
        self,
        joint_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(joint_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 4, joint_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(joint_dim)

    def forward(self, joint_tokens: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(
            query=joint_tokens,
            key=joint_tokens,
            value=joint_tokens
        )
        x = self.norm(joint_tokens + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class HeatmapAttentionLiftingNetwork(nn.Module):
    """
    Per-Joint Heatmap Attention-based lifting network.

    Key difference from GlobalAttentionLiftingNetwork:
    - Uses 16 heatmap tokens instead of 1 global token
    - K/V: [Backbone 64 + Heatmap 16] = 80 tokens
    """

    def __init__(
        self,
        num_joints: int = 16,
        joint_dim: int = 64,
        backbone_channels: int = 2048,
        hmd_dim: int = 9,
        num_heads: int = 4,
        num_self_attn_layers: int = 2,
        dropout: float = 0.1,
        use_confidence: bool = True
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.use_confidence = use_confidence

        # 2D coords → joint tokens
        input_dim = 3 if use_confidence else 2
        self.joint_embed = nn.Sequential(
            nn.Linear(input_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim)
        )

        # Positional embedding for joints
        self.joint_pos_embed = nn.Parameter(torch.randn(1, num_joints, joint_dim) * 0.02)

        # Heatmap + Backbone cross-attention
        self.heatmap_backbone_cross_attn = HeatmapBackboneCrossAttention(
            joint_dim=joint_dim,
            backbone_channels=backbone_channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # HMD cross-attention
        self.hmd_cross_attn = HMDCrossAttention(
            joint_dim=joint_dim,
            hmd_dim=hmd_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Joint self-attention layers
        self.self_attn_layers = nn.ModuleList([
            JointSelfAttention(joint_dim, num_heads, dropout)
            for _ in range(num_self_attn_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(joint_dim, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, 3)
        )

    def forward(
        self,
        coords_2d: Tensor,
        confidence: Tensor,
        backbone_feat: Tensor,
        hmd_info: Tensor,
        heatmap_tokens: Tensor
    ) -> Tuple[Tensor, dict]:
        """
        Args:
            coords_2d: [B, 16, 2] normalized 2D coordinates
            confidence: [B, 16] heatmap confidence
            backbone_feat: [B, 2048, 8, 8] backbone features
            hmd_info: [B, 9] HMD information
            heatmap_tokens: [B, 16, D] per-joint heatmap tokens

        Returns:
            pose_3d: [B, 16, 3]
            attn_info: dict with attention weights
        """
        B = coords_2d.size(0)

        # Combine coords and confidence
        if self.use_confidence:
            joint_input = torch.cat([coords_2d, confidence.unsqueeze(-1)], dim=-1)
        else:
            joint_input = coords_2d

        # Embed joints
        joint_tokens = self.joint_embed(joint_input)
        joint_tokens = joint_tokens + self.joint_pos_embed

        # Heatmap + Backbone cross-attention
        joint_tokens, backbone_heatmap_attn = self.heatmap_backbone_cross_attn(
            joint_tokens, backbone_feat, heatmap_tokens
        )
        # backbone_heatmap_attn: [B, 16, 80]

        # HMD cross-attention
        joint_tokens, hmd_attn = self.hmd_cross_attn(joint_tokens, hmd_info)

        # Self-attention layers
        for self_attn in self.self_attn_layers:
            joint_tokens = self_attn(joint_tokens)

        # Output projection
        pose_3d = self.output_proj(joint_tokens)

        # Extract attention info
        backbone_attn = backbone_heatmap_attn[:, :, :64]  # [B, 16, 64]
        heatmap_attn = backbone_heatmap_attn[:, :, 64:]   # [B, 16, 16]

        attn_info = {
            'backbone_attn': backbone_attn,
            'heatmap_attn': heatmap_attn,  # Joint-to-joint heatmap attention
            'hmd_attn': hmd_attn
        }

        return pose_3d, attn_info


@MODELS.register_module()
class CustomEgoposeHeatmapAttentionLiftingHead(BaseHead):
    """
    Per-Joint Heatmap Attention Lifting Head.

    Key improvement over Global Attention Lifting:
    - 16 heatmap tokens instead of 1 global token
    - Each joint's heatmap encoded as individual token
    - K/V: [Backbone 64 + Heatmap 16] = 80 tokens
    - 20% of K/V is heatmap info (vs 1.5% with global token)
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_out_channels: OptIntSeq = (256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4),
        deconv_stride_sizes: OptIntSeq = (2, 2),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1),
        # Attention params
        joint_dim: int = 64,
        num_heads: int = 4,
        num_self_attn_layers: int = 2,
        dropout: float = 0.1,
        detach_2d_coords: bool = True,
        # Loss configs
        loss: ConfigType = dict(type='KeypointMSELoss', loss_weight=1000),
        loss_pose_l2norm: ConfigType = dict(type='pose_l2norm', loss_weight=1.0),
        loss_cosine_similarity: ConfigType = dict(type='cosine_similarity', loss_weight=0.1),
        loss_limb_length: ConfigType = dict(type='limb_length', loss_weight=0.25),
        loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss', loss_weight=250),
        loss_hmd: ConfigType = dict(type='MSELoss', loss_weight=1.0),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_dim = joint_dim
        self.detach_2d_coords = detach_2d_coords

        # Build losses
        self.loss_module = MODELS.build(loss)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # Per-joint heatmap encoder (NEW)
        self.heatmap_encoder = PerJointHeatmapEncoder(
            num_joints=out_channels,
            output_dim=joint_dim
        )

        # Global encoder for Z (used in heatmap reconstruction)
        self.global_encoder = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, joint_dim),
            nn.LeakyReLU(0.2)
        )

        # EfficientHeatmapDecoder (for Z regularization)
        self.heatmap_decoder = EfficientHeatmapDecoder(
            num_classes=out_channels,
            heatmap_resolution=47,
            input_size=joint_dim
        )

        # Heatmap Attention-based lifting network
        self.lifting_network = HeatmapAttentionLiftingNetwork(
            num_joints=out_channels,
            joint_dim=joint_dim,
            backbone_channels=in_channels,
            hmd_dim=9,
            num_heads=num_heads,
            num_self_attn_layers=num_self_attn_layers,
            dropout=dropout,
            use_confidence=True
        )

        # Decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Build deconv layers
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length.')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
                layer_stride_sizes=deconv_stride_sizes,
            )
            in_channels_after_deconv = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()
            in_channels_after_deconv = in_channels

        # Conv layers
        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length.')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels_after_deconv,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
        else:
            self.conv_layers = nn.Identity()

        # Final layer
        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels_after_deconv,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Upsample to 47x47
        self.add_deconv_layers = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int],
                            layer_stride_sizes: Sequence[int]) -> nn.Module:
        layers = []
        for out_channels, kernel_size, stride_size in zip(
            layer_out_channels, layer_kernel_sizes, layer_stride_sizes
        ):
            if kernel_size == 4:
                padding, output_padding = 1, 0
            elif kernel_size == 3:
                padding, output_padding = 1, 1
            elif kernel_size == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size}')

            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_size,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Generate heatmaps from backbone features."""
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def forward_lifting(
        self,
        heatmaps: Tensor,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Forward pass for heatmap attention-based lifting.
        """
        # Extract per-joint heatmap tokens
        heatmap_tokens = self.heatmap_encoder(heatmaps.float())  # [B, 16, D]

        # Global Z for heatmap reconstruction
        global_z = self.global_encoder(heatmaps.float())  # [B, D]

        # Extract 2D coordinates
        coords_2d, confidence = soft_argmax_2d(heatmaps)

        # Detach 2D coords
        if self.detach_2d_coords:
            coords_2d_lift = coords_2d.detach()
            confidence_lift = confidence.detach()
        else:
            coords_2d_lift = coords_2d
            confidence_lift = confidence

        # Heatmap Attention-based lifting
        pose_3d, attn_info = self.lifting_network(
            coords_2d_lift, confidence_lift, backbone_feat, hmd_info, heatmap_tokens
        )

        return pose_3d, coords_2d, confidence, global_z, attn_info

    def decode(
        self,
        batch_heatmaps: Tensor,
        backbone_feat: Tensor,
        batch_data_samples: OptSampleList
    ) -> Tuple[InstanceList, Tensor]:
        """Decode predictions."""

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError('Decoder not set')

        # Get HMD info
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Lifting with heatmap tokens
        pose_3d, coords_2d, confidence, global_z, attn_info = self.forward_lifting(
            batch_heatmaps, backbone_feat, HMD_info.float()
        )

        # Reconstruct heatmaps from Z
        generated_heatmaps = self.heatmap_decoder(global_z)

        # HMD reconstruction from pose
        hmd_recons = self._compute_hmd_from_pose(pose_3d)

        # Decode 2D keypoints
        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_heatmaps, self.decoder.batch_decode)
            if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                batch_scores, batch_visibility = batch_scores
            else:
                batch_visibility = [None] * len(batch_keypoints)
        else:
            batch_output_np = to_numpy(batch_heatmaps, unzip=True)
            batch_keypoints = []
            batch_scores = []
            batch_visibility = []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        preds = []
        for i, (keypoints, scores, visibility) in enumerate(
            zip(batch_keypoints, batch_scores, batch_visibility)
        ):
            pred = InstanceData(
                keypoints=keypoints,
                keypoint_scores=scores,
                keypoint_3d=pose_3d[i:i+1],
                generated_heatmap=generated_heatmaps[i:i+1],
                hmd_recon=hmd_recons[i:i+1],
                coords_2d=coords_2d[i:i+1],
                confidence=confidence[i:i+1],
                heatmap_attn=attn_info['heatmap_attn'][i:i+1]
            )
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, pose_3d

    def _compute_hmd_from_pose(self, pose_3d: Tensor) -> Tensor:
        """Compute HMD info from 3D pose prediction."""
        head = pose_3d[:, 0]
        right_hand = pose_3d[:, 7]
        left_hand = pose_3d[:, 4]

        midpoint = (right_hand + left_hand) / 2
        z_axis = midpoint - head
        z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-6)

        hand_vector = right_hand - left_hand
        x_axis = torch.cross(z_axis, hand_vector, dim=1)
        x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-6)

        y_axis = torch.cross(z_axis, x_axis, dim=1)

        rotation_matrices = torch.stack((x_axis, y_axis, z_axis), dim=2)

        right_local = torch.bmm(
            rotation_matrices.transpose(1, 2),
            (right_hand - head).unsqueeze(2)
        ).squeeze(2)
        left_local = torch.bmm(
            rotation_matrices.transpose(1, 2),
            (left_hand - head).unsqueeze(2)
        ).squeeze(2)

        hand_distance = torch.norm(right_local - left_local, dim=1)
        right_distance = torch.norm(right_local, dim=1)
        left_distance = torch.norm(left_local, dim=1)

        hmd_recon = torch.cat([
            right_local, left_local,
            hand_distance.unsqueeze(1),
            right_distance.unsqueeze(1),
            left_distance.unsqueeze(1)
        ], dim=1)

        return hmd_recon

    def predict(
        self,
        feats: Features,
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {}
    ) -> Predictions:
        """Predict from features."""

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
            backbone_feat = _feats[-1]
        else:
            batch_heatmaps = self.forward(feats)
            backbone_feat = feats[-1]

        preds, _ = self.decode(batch_heatmaps, backbone_feat, batch_data_samples)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {}
    ) -> dict:
        """Calculate losses."""

        backbone_feat = feats[-1]
        pred_heatmaps = self.forward(feats)

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # Decode
        preds, pred_3d = self.decode(pred_heatmaps, backbone_feat, batch_data_samples)

        # Get reconstructed heatmaps and HMD
        pred_recon_heatmap = torch.cat([p.generated_heatmap for p in preds])
        pred_recon_hmd = torch.cat([p.hmd_recon for p in preds])

        # GT
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        pred_3d = pred_3d.view(-1, 16, 3)

        # Compute losses
        losses = dict()

        # 2D heatmap loss
        loss_2dkpt = self.loss_module(pred_heatmaps, gt_heatmaps, keypoint_weights)
        losses['loss_kpt'] = loss_2dkpt

        # 3D pose losses
        loss_pose_l2norm = self.loss_pose_l2norm_module(pred_3d, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pred_3d, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pred_3d, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # Heatmap reconstruction loss
        loss_heatmap_recon = self.loss_heatmap_recon_module(
            pred_recon_heatmap, gt_heatmaps, keypoint_weights
        )
        losses['loss_heatmap_recon'] = loss_heatmap_recon

        # HMD reconstruction loss
        loss_hmd = self.loss_hmd_module(
            pred_recon_hmd.double(), HMD_info.double()
        )
        losses['loss_hmd'] = loss_hmd

        # Accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            losses['acc_pose'] = torch.tensor(avg_acc, device=gt_heatmaps.device)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args, **kwargs):
        """Handle old version state dicts."""
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    k_new = k
            else:
                k_new = k
            state_dict[prefix + k_new] = v
