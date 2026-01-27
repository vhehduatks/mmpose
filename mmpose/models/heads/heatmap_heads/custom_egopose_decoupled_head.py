# Copyright (c) OpenMMLab. All rights reserved.
"""
Upper-Lower Decoupled Head (Option 1)

Key Insight:
    ViT v3: Upper Body 23.49mm (best!), Lower Body 67.19mm (worse)
    Baseline: Upper Body 29.42mm, Lower Body 53.31mm (better)

    → Use ViT v3 for Upper Body (HMD available for head/hands)
    → Use Baseline for Lower Body (no HMD info, Z-vector based)

Expected Result:
    Upper Body: 23.49mm (ViT v3)
    Lower Body: 53.31mm (Baseline)
    Full Body: (23.49 * 8 + 53.31 * 8) / 16 = 38.40mm
    → ~3mm better than Baseline (41.37mm)!

Architecture:
    Backbone feat [2048, 8, 8]
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  ┌─────────────┐ ┌─────────────┐
  │ Upper Body  │ │ Lower Body  │
  │ (ViT v3)    │ │ (Baseline)  │
  │             │ │             │
  │ Joint[8]    │ │ Deconv      │
  │ +HMD Attn   │ │ Heatmap[8]  │
  │ →pose[8,3]  │ │ Encoder→Z   │
  └─────────────┘ │ Decoder     │
        │         │ →pose[8,3]  │
        │         └─────────────┘
        │               │
        └───────┬───────┘
                │
          Concat → [16, 3]
                │
         (Optional) Refinement
                │
          3D Pose [16, 3]

Joint Indices:
    Upper Body (0-7): Head, Neck, R_Shoulder, R_Elbow, L_Wrist, L_Elbow, L_Shoulder, R_Wrist
    Lower Body (8-15): R_Knee, R_Ankle, R_Foot, L_Hip, L_Knee, L_Ankle, L_Foot, Pelvis
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

OptIntSeq = Optional[Sequence[int]]


# =============================================================================
# Upper Body Branch Components (ViT v3 style)
# =============================================================================

class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for spatial tokens."""

    def __init__(self, embed_dim: int, h: int = 8, w: int = 8):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(h * w, embed_dim)
        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w).flatten()
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1).flatten()

        div_term = torch.exp(torch.arange(0, embed_dim // 2, 2) *
                            -(math.log(10000.0) / (embed_dim // 2)))

        pe[:, 0::4] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 1::4] = torch.cos(x_pos.unsqueeze(1) * div_term)
        pe[:, 2::4] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, 3::4] = torch.cos(y_pos.unsqueeze(1) * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer with Pre-LayerNorm."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class PerJointHeatmapDecoder(nn.Module):
    """Per-Joint Heatmap Decoder for reconstruction regularization."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_joints: int = 8,  # Upper body only
        heatmap_size: int = 47,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, joint_tokens: Tensor) -> Tensor:
        B, N, D = joint_tokens.shape
        tokens_flat = joint_tokens.reshape(B * N, D)
        features = self.fc(tokens_flat)
        features = features.reshape(B * N, -1, 1, 1)
        heatmaps = self.upsample(features)
        heatmaps = heatmaps.reshape(B, N, self.heatmap_size, self.heatmap_size)
        return heatmaps


class HMDCrossAttention(nn.Module):
    """Cross-attention: Joint tokens query HMD tokens for 3D reference."""

    def __init__(
        self,
        embed_dim: int = 256,
        hmd_dim: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 3)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, joint_tokens: Tensor, hmd_info: Tensor) -> Tensor:
        B = joint_tokens.size(0)
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)
        joint_norm = self.norm(joint_tokens)
        attn_out, _ = self.cross_attn(
            query=joint_norm,
            key=hmd_tokens,
            value=hmd_tokens
        )
        return joint_tokens + self.dropout(attn_out)


class UpperBodyBranch(nn.Module):
    """
    Upper Body Branch using ViT v3 style.

    - 8 joint queries for upper body joints
    - Self-attention between spatial and joint tokens
    - HMD cross-attention for depth reference
    - Heatmap reconstruction for 2D regularization
    """

    def __init__(
        self,
        num_joints: int = 8,
        embed_dim: int = 256,
        backbone_channels: int = 2048,
        spatial_size: int = 8,
        heatmap_size: int = 47,
        hmd_dim: int = 9,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_heatmap_recon: bool = True
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.spatial_tokens_count = spatial_size * spatial_size
        self.use_heatmap_recon = use_heatmap_recon

        # Spatial projection
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        self.spatial_pos_enc = PositionalEncoding2D(embed_dim, spatial_size, spatial_size)

        # Joint queries for upper body (8 joints)
        self.joint_queries = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # Type embeddings
        self.spatial_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Heatmap reconstruction
        if use_heatmap_recon:
            self.heatmap_decoder = PerJointHeatmapDecoder(
                embed_dim=embed_dim,
                num_joints=num_joints,
                heatmap_size=heatmap_size
            )

        # HMD cross-attention (key for upper body!)
        self.hmd_cross_attn = HMDCrossAttention(
            embed_dim=embed_dim,
            hmd_dim=hmd_dim,
            num_heads=num_heads // 2,
            dropout=dropout
        )

        # 3D pose head
        self.head_3d = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)
        )

    def forward(
        self,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]
        Returns:
            upper_pose: [B, 8, 3]
            upper_heatmaps: [B, 8, 47, 47] or None
        """
        B = backbone_feat.size(0)

        # Spatial tokens
        spatial_feat = self.spatial_proj(backbone_feat)
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)
        spatial_tokens = self.spatial_pos_enc(spatial_tokens)
        spatial_tokens = spatial_tokens + self.spatial_type_embed

        # Joint tokens
        joint_tokens = self.joint_queries.expand(B, -1, -1).clone()
        joint_tokens = joint_tokens + self.joint_type_embed

        # Concat and self-attention
        tokens = torch.cat([spatial_tokens, joint_tokens], dim=1)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        # Extract joint tokens
        joint_tokens = tokens[:, -self.num_joints:, :]

        # Heatmap reconstruction
        upper_heatmaps = None
        if self.use_heatmap_recon:
            upper_heatmaps = self.heatmap_decoder(joint_tokens)

        # HMD cross-attention
        joint_tokens = self.hmd_cross_attn(joint_tokens, hmd_info)

        # 3D pose
        upper_pose = self.head_3d(joint_tokens)

        return upper_pose, upper_heatmaps


# =============================================================================
# Lower Body Branch Components (Baseline style)
# =============================================================================

class LowerBodyEncoder(nn.Module):
    """
    Encoder for lower body heatmaps → Z vector.
    Based on baseline's Encoder but for 8 joints only.
    """

    def __init__(self, num_joints: int = 8, output_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_joints, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(256, output_size)
        self.lrelu4 = nn.LeakyReLU(0.2)

    def forward(self, hm: Tensor) -> Tensor:
        """
        Args:
            hm: [B, 8, 47, 47] lower body heatmaps
        Returns:
            z: [B, 64]
        """
        hm = self.conv1(hm)
        hm = self.lrelu1(hm)
        hm = self.conv2(hm)
        hm = self.lrelu2(hm)
        hm = self.conv3(hm)
        hm = self.lrelu3(hm)

        hm_avgpool = self.avr_pool(hm).view(-1, 256)
        z = self.linear(hm_avgpool)
        z = self.lrelu4(z)

        return z


class LowerBodyPoseDecoder(nn.Module):
    """
    Decoder for Z → lower body 3D pose.
    Based on baseline's LinearModel but for 8 joints.
    """

    def __init__(
        self,
        input_size: int = 64,
        num_joints: int = 8,
        linear_size: int = 256,
        num_stage: int = 1,
        p_dropout: float = 0.3
    ):
        super().__init__()
        self.linear_size = linear_size
        self.num_stage = num_stage
        self.output_size = num_joints * 3

        self.w1 = nn.Linear(input_size, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # Residual blocks
        self.linear_stages = nn.ModuleList()
        for _ in range(num_stage):
            self.linear_stages.append(self._make_linear_block(linear_size, p_dropout))

        self.w2 = nn.Linear(linear_size, self.output_size)

    def _make_linear_block(self, linear_size: int, p_dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, 64]
        Returns:
            pose: [B, 8, 3]
        """
        y = self.w1(z)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for block in self.linear_stages:
            y = y + block(y)  # Residual connection

        y = self.w2(y)
        return y.view(-1, 8, 3)


class LowerBodyHeatmapDecoder(nn.Module):
    """Decoder for Z → lower body heatmaps (for reconstruction loss)."""

    def __init__(
        self,
        num_joints: int = 8,
        heatmap_size: int = 47,
        input_size: int = 64
    ):
        super().__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # Small version for efficiency
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_joints * 6 * 6),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_joints, num_joints * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_joints * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_joints * 4, num_joints * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_joints * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_joints * 2, num_joints, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_joints),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
            nn.Conv2d(num_joints, num_joints, kernel_size=3, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, 64]
        Returns:
            heatmaps: [B, 8, 47, 47]
        """
        B = z.size(0)
        x = self.decoder(z)
        x = x.view(B, self.num_joints, 6, 6)
        heatmaps = self.upsample(x)
        return heatmaps


class LowerBodyBranch(nn.Module):
    """
    Lower Body Branch using Baseline style.

    - Deconv → Heatmap[8] → Encoder → Z[64] → PoseDecoder → 3D pose[8, 3]
    - No HMD cross-attention (lower body has no HMD info)
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_joints: int = 8,
        heatmap_size: int = 47,
        z_size: int = 64,
        use_heatmap_recon: bool = True
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_heatmap_recon = use_heatmap_recon

        # Deconv layers (backbone → heatmap)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsample to heatmap size
        self.upsample_to_heatmap = nn.Sequential(
            nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final layer → 8 channel heatmap
        self.final_layer = nn.Conv2d(256, num_joints, kernel_size=1)

        # Encoder: Heatmap → Z
        self.encoder = LowerBodyEncoder(num_joints=num_joints, output_size=z_size)

        # Pose decoder: Z → 3D pose
        self.pose_decoder = LowerBodyPoseDecoder(
            input_size=z_size,
            num_joints=num_joints,
            linear_size=256,
            num_stage=1
        )

        # Heatmap decoder for reconstruction loss
        if use_heatmap_recon:
            self.heatmap_decoder = LowerBodyHeatmapDecoder(
                num_joints=num_joints,
                heatmap_size=heatmap_size,
                input_size=z_size
            )

    def forward(self, backbone_feat: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            backbone_feat: [B, 2048, 8, 8]
        Returns:
            lower_pose: [B, 8, 3]
            pred_heatmaps: [B, 8, 47, 47] predicted from deconv
            recon_heatmaps: [B, 8, 47, 47] reconstructed from Z (or None)
        """
        # Deconv → Heatmap
        x = self.deconv_layers(backbone_feat)
        x = self.upsample_to_heatmap(x)
        pred_heatmaps = self.final_layer(x)  # [B, 8, 47, 47]

        # Encoder → Z
        z = self.encoder(pred_heatmaps)  # [B, 64]

        # Pose decoder → 3D pose
        lower_pose = self.pose_decoder(z)  # [B, 8, 3]

        # Heatmap reconstruction
        recon_heatmaps = None
        if self.use_heatmap_recon:
            recon_heatmaps = self.heatmap_decoder(z)

        return lower_pose, pred_heatmaps, recon_heatmaps


# =============================================================================
# Main Decoupled Head
# =============================================================================

@MODELS.register_module()
class CustomEgoposeDecoupledHead(BaseHead):
    """
    Upper-Lower Decoupled Head.

    Combines ViT v3 for upper body (with HMD) and Baseline for lower body.

    Joint Indices:
        Upper Body (0-7): Head, Neck, R_Shoulder, R_Elbow, L_Wrist, L_Elbow, L_Shoulder, R_Wrist
        Lower Body (8-15): R_Knee, R_Ankle, R_Foot, L_Hip, L_Knee, L_Ankle, L_Foot, Pelvis

    Args:
        in_channels: Input channels from backbone (2048 for ResNet-101)
        out_channels: Total number of keypoints (16)
        embed_dim: Transformer embedding dimension for upper body (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_refinement: Whether to use final refinement layer (default: False)
    """

    _version = 1

    # Joint indices
    UPPER_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7]  # Head, Neck, R_Shoulder, R_Elbow, L_Wrist, L_Elbow, L_Shoulder, R_Wrist
    LOWER_JOINTS = [8, 9, 10, 11, 12, 13, 14, 15]  # R_Knee, R_Ankle, R_Foot, L_Hip, L_Knee, L_Ankle, L_Foot, Pelvis

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int = 16,
        # Upper body (ViT v3) params
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        heatmap_size: int = 47,
        # Lower body (Baseline) params
        z_size: int = 64,
        # Common params
        use_heatmap_recon: bool = True,
        use_refinement: bool = False,
        # Loss configs
        loss_upper_heatmap_recon: ConfigType = dict(type='KeypointMSELoss', loss_weight=500),
        loss_lower_heatmap: ConfigType = dict(type='KeypointMSELoss', loss_weight=500),
        loss_lower_heatmap_recon: ConfigType = dict(type='KeypointMSELoss', loss_weight=500),
        loss_pose_l2norm: ConfigType = dict(type='pose_l2norm', loss_weight=1.0),
        loss_cosine_similarity: ConfigType = dict(type='cosine_similarity', loss_weight=0.1),
        loss_limb_length: ConfigType = dict(type='limb_length', loss_weight=0.25),
        loss_hmd: ConfigType = dict(type='MSELoss', loss_weight=1.0),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heatmap_size = heatmap_size
        self.use_heatmap_recon = use_heatmap_recon
        self.use_refinement = use_refinement

        # Build losses
        self.loss_upper_heatmap_recon = MODELS.build(loss_upper_heatmap_recon) if use_heatmap_recon else None
        self.loss_lower_heatmap = MODELS.build(loss_lower_heatmap)
        self.loss_lower_heatmap_recon = MODELS.build(loss_lower_heatmap_recon) if use_heatmap_recon else None
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # Upper body branch (ViT v3 style)
        self.upper_branch = UpperBodyBranch(
            num_joints=8,
            embed_dim=embed_dim,
            backbone_channels=in_channels,
            spatial_size=8,
            heatmap_size=heatmap_size,
            hmd_dim=9,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_heatmap_recon=use_heatmap_recon
        )

        # Lower body branch (Baseline style)
        self.lower_branch = LowerBodyBranch(
            in_channels=in_channels,
            num_joints=8,
            heatmap_size=heatmap_size,
            z_size=z_size,
            use_heatmap_recon=use_heatmap_recon
        )

        # Optional refinement layer
        if use_refinement:
            self.refinement = nn.Sequential(
                nn.Linear(16 * 3, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 16 * 3),
            )

        # Decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='TruncNormal', layer='Linear', std=0.02)
        ]

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward pass - returns None as we use forward_decoupled."""
        return None

    def forward_decoupled(
        self,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> dict:
        """
        Forward pass for decoupled network.

        Args:
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]

        Returns:
            dict with:
                - pose_3d: [B, 16, 3]
                - upper_pose: [B, 8, 3]
                - lower_pose: [B, 8, 3]
                - upper_heatmaps: [B, 8, 47, 47] or None
                - lower_pred_heatmaps: [B, 8, 47, 47]
                - lower_recon_heatmaps: [B, 8, 47, 47] or None
        """
        # Upper body (ViT v3 with HMD)
        upper_pose, upper_heatmaps = self.upper_branch(backbone_feat, hmd_info)

        # Lower body (Baseline without HMD)
        lower_pose, lower_pred_heatmaps, lower_recon_heatmaps = self.lower_branch(backbone_feat)

        # Concatenate: [upper, lower] → [16, 3]
        pose_3d = torch.cat([upper_pose, lower_pose], dim=1)

        # Optional refinement
        if self.use_refinement:
            B = pose_3d.size(0)
            pose_flat = pose_3d.view(B, -1)
            pose_refined = pose_flat + self.refinement(pose_flat)
            pose_3d = pose_refined.view(B, 16, 3)

        return {
            'pose_3d': pose_3d,
            'upper_pose': upper_pose,
            'lower_pose': lower_pose,
            'upper_heatmaps': upper_heatmaps,
            'lower_pred_heatmaps': lower_pred_heatmaps,
            'lower_recon_heatmaps': lower_recon_heatmaps,
        }

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

    def decode(
        self,
        backbone_feat: Tensor,
        batch_data_samples: OptSampleList
    ) -> Tuple[InstanceList, Tensor]:
        """Decode predictions."""
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        outputs = self.forward_decoupled(backbone_feat, HMD_info.float())
        pose_3d = outputs['pose_3d']

        # HMD reconstruction
        hmd_recons = self._compute_hmd_from_pose(pose_3d)

        # Combine heatmaps for 2D decoding
        upper_heatmaps = outputs['upper_heatmaps']
        lower_pred_heatmaps = outputs['lower_pred_heatmaps']

        if upper_heatmaps is not None:
            full_heatmaps = torch.cat([upper_heatmaps, lower_pred_heatmaps], dim=1)
        else:
            # Use lower_pred_heatmaps doubled if no upper heatmaps
            full_heatmaps = lower_pred_heatmaps

        # Decode 2D keypoints
        B = pose_3d.shape[0]
        if full_heatmaps is not None and self.decoder is not None:
            if self.decoder.support_batch_decoding:
                batch_keypoints, batch_scores = self.decoder.batch_decode(full_heatmaps)
                if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                    batch_scores, batch_visibility = batch_scores
                else:
                    batch_visibility = [None] * len(batch_keypoints)
            else:
                batch_output_np = to_numpy(full_heatmaps, unzip=True)
                batch_keypoints = []
                batch_scores = []
                batch_visibility = []
                for out in batch_output_np:
                    kpts, scores = self.decoder.decode(out)
                    batch_keypoints.append(kpts)
                    if isinstance(scores, tuple):
                        batch_scores.append(scores[0])
                        batch_visibility.append(scores[1])
                    else:
                        batch_scores.append(scores)
                        batch_visibility.append(None)
        else:
            pose_2d = pose_3d[:, :, :2]
            batch_keypoints = pose_2d.detach().cpu().numpy() * 256
            batch_scores = [np.ones((16,)) for _ in range(B)]
            batch_visibility = [None] * B

        preds = []
        for i in range(B):
            if isinstance(batch_keypoints, np.ndarray):
                kpts = batch_keypoints[i:i+1]
            else:
                kpts = batch_keypoints[i]
                if kpts.ndim == 2:
                    kpts = kpts[np.newaxis, ...]

            if isinstance(batch_scores, np.ndarray):
                scores = batch_scores[i:i+1]
            else:
                scores = batch_scores[i]
                if isinstance(scores, np.ndarray) and scores.ndim == 1:
                    scores = scores[np.newaxis, ...]

            # Create generated heatmap for consistency with baseline
            if full_heatmaps is not None:
                gen_heatmap = full_heatmaps[i:i+1].detach().cpu().numpy()
            else:
                gen_heatmap = np.zeros((1, 16, self.heatmap_size, self.heatmap_size))

            pred = InstanceData(
                keypoints=kpts,
                keypoint_scores=scores,
                keypoint_3d=pose_3d[i:i+1].detach().cpu().numpy(),
                generated_heatmap=gen_heatmap,
                hmd_recon=hmd_recons[i:i+1].detach().cpu().numpy()
            )
            if batch_visibility[i] is not None:
                pred.keypoints_visible = batch_visibility[i]
            preds.append(pred)

        return preds, pose_3d

    def predict(
        self,
        feats: Features,
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {}
    ) -> Predictions:
        """Predict from features."""
        backbone_feat = feats[-1]
        preds, _ = self.decode(backbone_feat, batch_data_samples)

        if test_cfg.get('output_heatmaps', False):
            HMD_info = torch.cat([
                d.gt_instance_labels.hmd_info for d in batch_data_samples
            ])
            outputs = self.forward_decoupled(backbone_feat, HMD_info.float())
            upper_heatmaps = outputs['upper_heatmaps']
            lower_pred_heatmaps = outputs['lower_pred_heatmaps']
            if upper_heatmaps is not None:
                full_heatmaps = torch.cat([upper_heatmaps, lower_pred_heatmaps], dim=1)
                pred_fields = [PixelData(heatmaps=hm) for hm in full_heatmaps.detach()]
                return preds, pred_fields

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {}
    ) -> dict:
        """Calculate losses."""
        backbone_feat = feats[-1]

        # Get HMD info and GT
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])
        gt_heatmaps = torch.stack([
            d.gt_fields.heatmaps for d in batch_data_samples
        ])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # Forward
        outputs = self.forward_decoupled(backbone_feat, HMD_info.float())
        pose_3d = outputs['pose_3d']

        # HMD reconstruction
        hmd_recon = self._compute_hmd_from_pose(pose_3d)

        losses = dict()

        # ===== 3D Pose Losses (full body) =====
        pose_3d = pose_3d.view(-1, 16, 3)
        loss_pose_l2norm = self.loss_pose_l2norm_module(pose_3d, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pose_3d, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pose_3d, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # ===== HMD Loss =====
        loss_hmd = self.loss_hmd_module(hmd_recon.double(), HMD_info.double())
        losses['loss_hmd'] = loss_hmd

        # ===== Upper Body Heatmap Reconstruction Loss =====
        if self.use_heatmap_recon and outputs['upper_heatmaps'] is not None:
            gt_upper_heatmaps = gt_heatmaps[:, self.UPPER_JOINTS, :, :]
            upper_keypoint_weights = keypoint_weights[:, self.UPPER_JOINTS]

            loss_upper_heatmap_recon = self.loss_upper_heatmap_recon(
                outputs['upper_heatmaps'], gt_upper_heatmaps, upper_keypoint_weights
            )
            losses['loss_upper_hm_recon'] = loss_upper_heatmap_recon

        # ===== Lower Body Heatmap Loss =====
        gt_lower_heatmaps = gt_heatmaps[:, self.LOWER_JOINTS, :, :]
        lower_keypoint_weights = keypoint_weights[:, self.LOWER_JOINTS]

        loss_lower_heatmap = self.loss_lower_heatmap(
            outputs['lower_pred_heatmaps'], gt_lower_heatmaps, lower_keypoint_weights
        )
        losses['loss_lower_hm'] = loss_lower_heatmap

        if self.use_heatmap_recon and outputs['lower_recon_heatmaps'] is not None:
            loss_lower_heatmap_recon = self.loss_lower_heatmap_recon(
                outputs['lower_recon_heatmaps'], gt_lower_heatmaps, lower_keypoint_weights
            )
            losses['loss_lower_hm_recon'] = loss_lower_heatmap_recon

        # ===== Accuracy =====
        if train_cfg.get('compute_acc', True):
            # Upper body accuracy
            if outputs['upper_heatmaps'] is not None:
                _, upper_acc, _ = pose_pck_accuracy(
                    output=to_numpy(outputs['upper_heatmaps']),
                    target=to_numpy(gt_upper_heatmaps),
                    mask=to_numpy(upper_keypoint_weights) > 0
                )
                losses['acc_upper'] = torch.tensor(upper_acc, device=gt_heatmaps.device)

            # Lower body accuracy
            _, lower_acc, _ = pose_pck_accuracy(
                output=to_numpy(outputs['lower_pred_heatmaps']),
                target=to_numpy(gt_lower_heatmaps),
                mask=to_numpy(lower_keypoint_weights) > 0
            )
            losses['acc_lower'] = torch.tensor(lower_acc, device=gt_heatmaps.device)

            # Full body accuracy (average)
            if outputs['upper_heatmaps'] is not None:
                losses['acc_pose'] = (losses['acc_upper'] + losses['acc_lower']) / 2
            else:
                losses['acc_pose'] = losses['acc_lower']

        return losses
