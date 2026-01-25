# Copyright (c) OpenMMLab. All rights reserved.
"""
Global Context Enhanced Attention Lifting Head

Key improvement over Attention Lifting v1:
- Adds Global Heatmap Token to Cross-Attention K/V
- Each joint queries both spatial backbone features AND global context
- Combines Baseline's proven Encoder with Attention mechanism

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
      soft_argmax   Encoder                           │
           │         │                                │
           ▼         ▼                                │
      2D coords   Z [64] ──── Global Token [1, D] ────┤
      [16, 2]                                         │
           │                                          │
           ▼                                          │
      Joint embed [16, D]                             │
           │                                          │
           └──────── Cross-Attention ─────────────────┘
                     Q: joints [16, D]
                     K/V: [backbone + global] [65, D]
                              │
                              ▼
                     Depth-aware joints [16, D]
                              │
                              + HMD Cross-Attention
                              │  (Q: joints, K/V: HMD 3 tokens)
                              ▼
                        Self-Attention
                       (관절 간 관계)
                              │
                              ▼
                        3D Pose [16, 3]

Expected improvements:
- Global context from Baseline's Encoder (proven effective)
- Each joint can attend to global + spatial features
- Attention learns when to use global vs local context
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

    Args:
        heatmaps: [B, K, H, W] heatmap tensor
        temperature: softmax temperature (lower = sharper)

    Returns:
        coords: [B, K, 2] normalized coordinates (0~1)
        confidence: [B, K] max values as confidence
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)  # [B, K, H*W]

    # Softmax over spatial dimensions
    heatmaps_soft = F.softmax(heatmaps_flat / temperature, dim=-1)  # [B, K, H*W]

    # Create coordinate grids (normalized 0~1)
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    # Create meshgrid and flatten
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx_flat = xx.flatten()  # [H*W]
    yy_flat = yy.flatten()  # [H*W]

    # Compute expected coordinates
    x = (heatmaps_soft * xx_flat.view(1, 1, -1)).sum(dim=-1)  # [B, K]
    y = (heatmaps_soft * yy_flat.view(1, 1, -1)).sum(dim=-1)  # [B, K]

    coords = torch.stack([x, y], dim=-1)  # [B, K, 2]

    # Confidence = max value of heatmap
    confidence = heatmaps.view(B, K, -1).max(dim=-1)[0]  # [B, K]

    return coords, confidence


class HeatmapEncoder(nn.Module):
    """
    Encode heatmap to latent vector Z (Global Context).

    This is the same encoder used in Baseline - proven to capture
    global relationships between all joints.
    """

    def __init__(self, num_classes: int = 16, output_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(256, output_size)
        self.lrelu4 = nn.LeakyReLU(0.2)

    def forward(self, hm: Tensor) -> Tensor:
        hm = self.lrelu1(self.conv1(hm))
        hm = self.lrelu2(self.conv2(hm))
        hm = self.lrelu3(self.conv3(hm))
        hm = self.avr_pool(hm).view(-1, 256)
        z = self.lrelu4(self.linear(hm))
        return z


class GlobalBackboneCrossAttention(nn.Module):
    """
    Cross-attention with Global Heatmap Token.

    Key improvement: K/V includes both spatial backbone tokens AND global context token.

    Q: Joint tokens [B, 16, D] - "이 2D 위치의 depth는? + 전체 context에서 나의 위치는?"
    K/V: [Backbone tokens [B, 64, D] + Global Token [B, 1, D]] = [B, 65, D]

    The attention can learn:
    - When to focus on local spatial features (occluded joints → nearby visible joints)
    - When to use global context (ambiguous depth → overall body configuration)
    """

    def __init__(
        self,
        joint_dim: int = 64,
        backbone_channels: int = 2048,
        backbone_spatial: int = 64,  # 8x8
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

        # Global token projection (from Z)
        self.global_proj = nn.Sequential(
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
        global_z: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            joint_tokens: [B, 16, D] joint queries
            backbone_feat: [B, 2048, 8, 8] backbone features
            global_z: [B, D] global context from HeatmapEncoder

        Returns:
            depth_aware_joints: [B, 16, D]
            attn_weights: [B, 16, 65] attention map (64 spatial + 1 global)
        """
        B = joint_tokens.size(0)

        # Backbone → spatial tokens [B, 64, D]
        backbone_tokens = self.backbone_proj(backbone_feat)  # [B, D*2, 8, 8]
        backbone_tokens = backbone_tokens.flatten(2).transpose(1, 2)  # [B, 64, D*2]
        backbone_kv = self.backbone_to_kv(backbone_tokens)  # [B, 64, D]

        # Global token [B, 1, D]
        global_token = self.global_proj(global_z).unsqueeze(1)  # [B, 1, D]

        # Concatenate: [B, 65, D] = [backbone spatial + global context]
        combined_kv = torch.cat([backbone_kv, global_token], dim=1)

        # Cross attention: joints query both spatial and global
        attn_out, attn_weights = self.cross_attn(
            query=joint_tokens,  # [B, 16, D]
            key=combined_kv,     # [B, 65, D]
            value=combined_kv
        )
        # attn_weights: [B, 16, 65] - last column is global attention!

        # Residual + norm
        out = self.norm(joint_tokens + self.dropout(attn_out))

        return out, attn_weights


class HMDCrossAttention(nn.Module):
    """
    Cross-attention: Joint tokens query HMD tokens for 3D reference.

    Q: Joint tokens [B, 16, D]
    K/V: HMD tokens [B, 3, D] - (head, right_hand, left_hand)

    Effect: 손 관절 → 손 HMD, 몸통 → head HMD
    """

    def __init__(
        self,
        joint_dim: int = 64,
        hmd_dim: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # HMD → 3 tokens
        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Linear(joint_dim * 2, joint_dim * 3)  # 3 tokens
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
        hmd_info: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            joint_tokens: [B, 16, D]
            hmd_info: [B, 9]

        Returns:
            hmd_aware_joints: [B, 16, D]
            attn_weights: [B, 16, 3]
        """
        B = joint_tokens.size(0)

        # HMD → 3 tokens [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention
        attn_out, attn_weights = self.cross_attn(
            query=joint_tokens,
            key=hmd_tokens,
            value=hmd_tokens
        )

        # Residual + norm
        out = self.norm(joint_tokens + self.dropout(attn_out))

        return out, attn_weights


class JointSelfAttention(nn.Module):
    """
    Self-attention among joint tokens for structural relationships.

    Effect:
    - 왼팔 ↔ 오른팔 대칭 관계
    - 부모-자식 관계 (어깨→팔꿈치→손목)
    - 가려진 관절이 보이는 관절 참조
    """

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

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 4, joint_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(joint_dim)

    def forward(self, joint_tokens: Tensor) -> Tensor:
        """
        Args:
            joint_tokens: [B, 16, D]

        Returns:
            refined_joints: [B, 16, D]
        """
        # Self attention
        attn_out, _ = self.self_attn(
            query=joint_tokens,
            key=joint_tokens,
            value=joint_tokens
        )
        x = self.norm(joint_tokens + self.dropout(attn_out))

        # FFN
        x = self.norm2(x + self.ffn(x))

        return x


class GlobalAttentionLiftingNetwork(nn.Module):
    """
    Global Context Enhanced Attention-based lifting network.

    Key difference from AttentionLiftingNetwork:
    - GlobalBackboneCrossAttention: K/V includes global heatmap token
    - Each joint can attend to both spatial backbone features AND global context

    Pipeline:
    1. 2D coords → Joint embed
    2. Global Backbone Cross-Attention (spatial + global depth query)
    3. HMD Cross-Attention (3D reference)
    4. Joint Self-Attention (structural relations)
    5. Output projection → 3D pose
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
        input_dim = 3 if use_confidence else 2  # (x, y, conf) or (x, y)
        self.joint_embed = nn.Sequential(
            nn.Linear(input_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim)
        )

        # Positional embedding for joints
        self.joint_pos_embed = nn.Parameter(torch.randn(1, num_joints, joint_dim) * 0.02)

        # Global Backbone cross-attention (NEW: includes global token)
        self.global_backbone_cross_attn = GlobalBackboneCrossAttention(
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
        global_z: Tensor
    ) -> Tuple[Tensor, dict]:
        """
        Args:
            coords_2d: [B, 16, 2] normalized 2D coordinates
            confidence: [B, 16] heatmap confidence
            backbone_feat: [B, 2048, 8, 8] backbone features
            hmd_info: [B, 9] HMD information
            global_z: [B, 64] global context from HeatmapEncoder

        Returns:
            pose_3d: [B, 16, 3]
            attn_info: dict with attention weights for visualization
        """
        B = coords_2d.size(0)

        # Combine coords and confidence
        if self.use_confidence:
            joint_input = torch.cat([coords_2d, confidence.unsqueeze(-1)], dim=-1)  # [B, 16, 3]
        else:
            joint_input = coords_2d  # [B, 16, 2]

        # Embed joints
        joint_tokens = self.joint_embed(joint_input)  # [B, 16, D]
        joint_tokens = joint_tokens + self.joint_pos_embed

        # Global Backbone cross-attention (spatial + global)
        joint_tokens, backbone_attn = self.global_backbone_cross_attn(
            joint_tokens, backbone_feat, global_z
        )
        # backbone_attn: [B, 16, 65] - last column is global attention!

        # HMD cross-attention (3D reference)
        joint_tokens, hmd_attn = self.hmd_cross_attn(joint_tokens, hmd_info)

        # Self-attention layers
        for self_attn in self.self_attn_layers:
            joint_tokens = self_attn(joint_tokens)

        # Output projection
        pose_3d = self.output_proj(joint_tokens)  # [B, 16, 3]

        # Extract global attention for analysis
        global_attn = backbone_attn[:, :, -1]  # [B, 16] - how much each joint attended to global

        attn_info = {
            'backbone_attn': backbone_attn[:, :, :-1],  # [B, 16, 64] - spatial attention
            'global_attn': global_attn,  # [B, 16] - global context attention
            'hmd_attn': hmd_attn  # [B, 16, 3] - which HMD (head/right/left)
        }

        return pose_3d, attn_info


@MODELS.register_module()
class CustomEgoposeGlobalAttentionLiftingHead(BaseHead):
    """
    Global Context Enhanced Attention Lifting Head.

    Key improvement over Attention Lifting v1:
    - Adds Global Heatmap Token to Cross-Attention K/V
    - Each joint queries both spatial backbone features AND global context
    - Combines Baseline's proven Encoder with Attention mechanism

    The global token provides:
    - Overall body configuration context
    - Inter-joint relationships (already encoded by HeatmapEncoder)
    - Disambiguation for occluded or ambiguous joints

    Args:
        in_channels: Input channels from backbone (2048 for ResNet-101)
        out_channels: Number of keypoints (16)
        joint_dim: Dimension of joint tokens (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_self_attn_layers: Number of self-attention layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        detach_2d_coords: Whether to detach 2D coords from gradient (default: True)
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

        # Heatmap encoder (for global context Z)
        self.encoder = HeatmapEncoder(num_classes=out_channels, output_size=joint_dim)

        # EfficientHeatmapDecoder (for Z regularization)
        self.heatmap_decoder = EfficientHeatmapDecoder(
            num_classes=out_channels,
            heatmap_resolution=47,
            input_size=joint_dim
        )

        # Global Attention-based lifting network
        self.lifting_network = GlobalAttentionLiftingNetwork(
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
        Forward pass for global attention-based lifting.

        Args:
            heatmaps: [B, 16, 47, 47]
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]

        Returns:
            pose_3d: [B, 16, 3]
            coords_2d: [B, 16, 2]
            confidence: [B, 16]
            global_z: [B, 64] - global context
            attn_info: attention weights
        """
        # Extract global context from heatmap (Baseline's proven encoder)
        global_z = self.encoder(heatmaps.float())  # [B, 64]

        # Extract 2D coordinates
        coords_2d, confidence = soft_argmax_2d(heatmaps)

        # Detach 2D coords (role separation: 2D loss doesn't affect lifting)
        if self.detach_2d_coords:
            coords_2d_lift = coords_2d.detach()
            confidence_lift = confidence.detach()
        else:
            coords_2d_lift = coords_2d
            confidence_lift = confidence

        # Global Attention-based lifting
        pose_3d, attn_info = self.lifting_network(
            coords_2d_lift, confidence_lift, backbone_feat, hmd_info, global_z
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

        # Lifting with global context
        pose_3d, coords_2d, confidence, global_z, attn_info = self.forward_lifting(
            batch_heatmaps, backbone_feat, HMD_info.float()
        )

        # Reconstruct heatmaps from Z (for regularization)
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
                global_attn=attn_info['global_attn'][i:i+1]  # Store global attention for analysis
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

        # Heatmap reconstruction loss (Z regularization)
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
