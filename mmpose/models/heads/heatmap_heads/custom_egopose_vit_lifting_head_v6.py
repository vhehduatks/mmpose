# Copyright (c) OpenMMLab. All rights reserved.
"""
ViT-Style Lifting Head v6 (Small Dataset Optimized)

Key innovations for small dataset (EgoPose ~210K):
- SPT (Shifted Patch Tokenization): Locality inductive bias injection
- LSA (Locality Self-Attention): Learnable temperature + diagonal masking
- Reduced model size: embed_dim=128, num_layers=2, mlp_ratio=2.0
- Depth-wise Conv embedding: Local feature enhancement

References:
- Vision Transformer for Small-Size Datasets (AAAI 2022)
- Depth-Wise Convolutions in ViTs (Neural Networks 2024)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Backbone feat [2048, 8, 8]                                 │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  SPT (Shifted Patch Tokenization)       │                │
    │  │  5-way shift → locality bias            │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  Depth-wise Conv Embedding              │                │
    │  │  Local feature enhancement              │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  Spatial Tokens [64, D] + Joint Queries [16, D]            │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  LSA (Locality Self-Attention) × 2      │                │
    │  │  Learnable temp + diagonal mask         │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  Joint Tokens [16, D]                                       │
    │         ├── Heatmap Decoder (Reconstruction)                │
    │         ↓                                                   │
    │  HMD Cross-Attention                                        │
    │         ↓                                                   │
    │  3D Pose [16, 3]                                            │
    └─────────────────────────────────────────────────────────────┘
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


class ShiftedPatchTokenization(nn.Module):
    """
    Shifted Patch Tokenization (SPT) for locality inductive bias.

    Shifts input feature maps in 4 directions and concatenates with original,
    effectively enlarging the receptive field and injecting locality bias.

    Input: [B, C, H, W]
    Output: [B, embed_dim, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        shift_size: int = 1
    ):
        super().__init__()
        self.shift_size = shift_size

        # 5 directions (original + 4 shifts) concatenated
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 5, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] backbone features

        Returns:
            tokens: [B, embed_dim, H, W] shifted patch tokens
        """
        B, C, H, W = x.shape
        s = self.shift_size

        # 4-way shift with padding
        x_left = F.pad(x, (s, 0, 0, 0))[:, :, :, :W]      # shift left
        x_right = F.pad(x, (0, s, 0, 0))[:, :, :, s:]     # shift right
        x_up = F.pad(x, (0, 0, s, 0))[:, :, :H, :]        # shift up
        x_down = F.pad(x, (0, 0, 0, s))[:, :, s:, :]      # shift down

        # Concatenate all 5 directions: [B, 5C, H, W]
        x_concat = torch.cat([x, x_left, x_right, x_up, x_down], dim=1)

        # Project to embed_dim: [B, embed_dim, H, W]
        return self.proj(x_concat)


class DepthWiseConvEmbedding(nn.Module):
    """
    Depth-wise Convolution Token Embedding.

    Adds CNN's local inductive bias to the tokenization process.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()

        # Depth-wise conv for local feature extraction
        self.dwconv = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
            bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, embed_dim, H, W]
        Returns:
            x: [B, embed_dim, H, W] with local features enhanced
        """
        return self.act(self.norm(self.dwconv(x))) + x  # Residual


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


class LocalitySelfAttention(nn.Module):
    """
    Locality Self-Attention (LSA) for small datasets.

    Key innovations:
    1. Learnable temperature: Controls attention sharpness
    2. Diagonal masking: Removes self-relation, forces attention to neighbors

    Reference: Vision Transformer for Small-Size Datasets (AAAI 2022)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        init_temperature: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Key innovation 1: Learnable temperature (init to sharp attention)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * init_temperature)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, use_diagonal_mask: bool = True) -> Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            use_diagonal_mask: Whether to mask diagonal (self-attention)

        Returns:
            x: [B, N, D] output tokens
        """
        B, N, D = x.shape

        # QKV projection: [B, N, 3D] → 3 × [B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Key innovation 1: Learnable temperature (sharper attention)
        attn = attn / self.temperature.clamp(min=0.1)

        # Key innovation 2: Diagonal masking (remove self-relation)
        if use_diagonal_mask:
            diag_mask = torch.eye(N, device=x.device, dtype=torch.bool)
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn = attn.masked_fill(diag_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention: [B, num_heads, N, head_dim]
        out = attn @ v

        # Reshape and project: [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LSAEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Locality Self-Attention.

    Uses Pre-LayerNorm for stability.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        init_temperature: float = 0.5
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LocalitySelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            init_temperature=init_temperature
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
        # Pre-LayerNorm LSA
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + self.dropout1(attn_out)

        # Pre-LayerNorm MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PerJointHeatmapDecoderSmall(nn.Module):
    """
    Smaller Heatmap Decoder for v6 (reduced params).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_joints: int = 16,
        heatmap_size: int = 47,
        hidden_dim: int = 128  # Reduced from 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # Smaller MLP
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Lighter upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),

            nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, joint_tokens: Tensor) -> Tensor:
        B, N, D = joint_tokens.shape

        tokens_flat = joint_tokens.reshape(B * N, D)
        features = self.fc(tokens_flat)
        features = features.reshape(B * N, -1, 1, 1)

        heatmaps = self.upsample(features)
        heatmaps = heatmaps.reshape(B, N, self.heatmap_size, self.heatmap_size)

        return heatmaps


class HMDCrossAttentionSmall(nn.Module):
    """
    Smaller HMD Cross-Attention for v6.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hmd_dim: int = 9,
        num_heads: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 3)
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

        hmd_tokens = self.hmd_embed(hmd_info).reshape(B, 3, -1)

        joint_norm = self.norm(joint_tokens)
        attn_out, _ = self.cross_attn(
            query=joint_norm,
            key=hmd_tokens,
            value=hmd_tokens
        )

        return joint_tokens + self.dropout(attn_out)


class ViTLiftingNetworkV6(nn.Module):
    """
    ViT-Style Lifting Network v6 (Small Dataset Optimized).

    Key features:
    - SPT (Shifted Patch Tokenization)
    - Depth-wise Conv embedding
    - LSA (Locality Self-Attention)
    - Reduced model size
    """

    def __init__(
        self,
        num_joints: int = 16,
        embed_dim: int = 128,          # Reduced from 256
        backbone_channels: int = 2048,
        spatial_size: int = 8,
        heatmap_size: int = 47,
        hmd_dim: int = 9,
        num_heads: int = 4,            # Reduced from 8
        num_layers: int = 2,           # Reduced from 4
        mlp_ratio: float = 2.0,        # Reduced from 4.0
        dropout: float = 0.2,          # Increased from 0.1
        init_temperature: float = 0.5,
        use_hmd: bool = True,
        use_heatmap_recon: bool = True,
        use_spt: bool = True,
        use_dwconv: bool = True,
        use_lsa: bool = True
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.spatial_tokens_count = spatial_size * spatial_size
        self.use_hmd = use_hmd
        self.use_heatmap_recon = use_heatmap_recon
        self.use_spt = use_spt
        self.use_dwconv = use_dwconv
        self.use_lsa = use_lsa

        # 1. Tokenization: SPT or standard projection
        if use_spt:
            self.spatial_proj = ShiftedPatchTokenization(
                in_channels=backbone_channels,
                embed_dim=embed_dim,
                shift_size=1
            )
        else:
            self.spatial_proj = nn.Sequential(
                nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )

        # 2. Depth-wise Conv embedding (optional)
        if use_dwconv:
            self.dwconv_embed = DepthWiseConvEmbedding(embed_dim, kernel_size=3)

        # 3. Positional encoding
        self.spatial_pos_enc = PositionalEncoding2D(embed_dim, spatial_size, spatial_size)

        # 4. Learnable Joint Queries
        self.joint_queries = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # 5. Type embeddings
        self.spatial_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 6. Transformer Encoder (LSA or standard)
        if use_lsa:
            self.transformer_layers = nn.ModuleList([
                LSAEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    init_temperature=init_temperature
                )
                for _ in range(num_layers)
            ])
        else:
            # Standard transformer (fallback)
            from .custom_egopose_vit_lifting_head import TransformerEncoderLayer
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(embed_dim)

        # 7. Heatmap Reconstruction Decoder (smaller)
        if use_heatmap_recon:
            self.heatmap_decoder = PerJointHeatmapDecoderSmall(
                embed_dim=embed_dim,
                num_joints=num_joints,
                heatmap_size=heatmap_size,
                hidden_dim=128
            )

        # 8. HMD Cross-Attention (smaller)
        if use_hmd:
            self.hmd_cross_attn = HMDCrossAttentionSmall(
                embed_dim=embed_dim,
                hmd_dim=hmd_dim,
                num_heads=num_heads // 2,
                dropout=dropout
            )

        # 9. 3D Pose Head (smaller)
        self.head_3d = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.joint_queries, std=0.02)
        nn.init.zeros_(self.spatial_type_embed)
        nn.init.zeros_(self.joint_type_embed)

    def forward(
        self,
        backbone_feat: Tensor,
        hmd_info: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], dict]:
        """
        Args:
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]

        Returns:
            pose_3d: [B, 16, 3]
            recon_heatmaps: [B, 16, 47, 47] or None
            info: dict
        """
        B = backbone_feat.size(0)

        # 1. SPT Tokenization: [B, D, 8, 8]
        spatial_feat = self.spatial_proj(backbone_feat)

        # 2. Depth-wise Conv (optional)
        if self.use_dwconv:
            spatial_feat = self.dwconv_embed(spatial_feat)

        # Flatten to tokens: [B, 64, D]
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)

        # Add positional encoding
        spatial_tokens = self.spatial_pos_enc(spatial_tokens)
        spatial_tokens = spatial_tokens + self.spatial_type_embed

        # 3. Joint queries: [B, 16, D]
        joint_tokens = self.joint_queries.expand(B, -1, -1).clone()
        joint_tokens = joint_tokens + self.joint_type_embed

        # 4. Concatenate: [B, 80, D]
        tokens = torch.cat([spatial_tokens, joint_tokens], dim=1)

        # 5. LSA layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)

        # 6. Extract joint tokens
        joint_tokens = tokens[:, -self.num_joints:, :]

        # 7. Heatmap Reconstruction
        recon_heatmaps = None
        if self.use_heatmap_recon:
            recon_heatmaps = self.heatmap_decoder(joint_tokens)

        # 8. HMD Cross-Attention
        if self.use_hmd and hmd_info is not None:
            joint_tokens = self.hmd_cross_attn(joint_tokens, hmd_info)

        # 9. 3D Pose
        pose_3d = self.head_3d(joint_tokens)

        info = {
            'spatial_tokens': spatial_tokens,
            'joint_tokens': joint_tokens
        }

        return pose_3d, recon_heatmaps, info


@MODELS.register_module()
class CustomEgoposeViTLiftingHeadV6(BaseHead):
    """
    ViT-Style Lifting Head v6 (Small Dataset Optimized).

    Key features for small datasets:
    - SPT (Shifted Patch Tokenization): Locality inductive bias
    - LSA (Locality Self-Attention): Learnable temperature + diagonal masking
    - Reduced model: embed_dim=128, num_layers=2, mlp_ratio=2.0
    - Higher dropout: 0.2 for regularization

    Args:
        in_channels: Input channels from backbone (2048)
        out_channels: Number of keypoints (16)
        embed_dim: Transformer embedding dim (default: 128, reduced)
        num_heads: Attention heads (default: 4, reduced)
        num_layers: Transformer layers (default: 2, reduced)
        mlp_ratio: MLP ratio (default: 2.0, reduced)
        dropout: Dropout rate (default: 0.2, increased)
        init_temperature: LSA temperature init (default: 0.5)
        use_spt: Use Shifted Patch Tokenization (default: True)
        use_dwconv: Use Depth-wise Conv embedding (default: True)
        use_lsa: Use Locality Self-Attention (default: True)
    """

    _version = 6

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        # v6 small model params
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        heatmap_size: int = 47,
        init_temperature: float = 0.5,
        # v6 specific flags
        use_spt: bool = True,
        use_dwconv: bool = True,
        use_lsa: bool = True,
        use_hmd: bool = True,
        use_heatmap_recon: bool = True,
        # Loss configs
        loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss', loss_weight=500),
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
        self.embed_dim = embed_dim
        self.heatmap_size = heatmap_size
        self.use_heatmap_recon = use_heatmap_recon

        # Build losses
        self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon) if use_heatmap_recon else None
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # ViT Lifting Network v6
        self.lifting_network = ViTLiftingNetworkV6(
            num_joints=out_channels,
            embed_dim=embed_dim,
            backbone_channels=in_channels,
            spatial_size=8,
            heatmap_size=heatmap_size,
            hmd_dim=9,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            init_temperature=init_temperature,
            use_hmd=use_hmd,
            use_heatmap_recon=use_heatmap_recon,
            use_spt=use_spt,
            use_dwconv=use_dwconv,
            use_lsa=use_lsa
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
        return None

    def forward_lifting(
        self,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], dict]:
        return self.lifting_network(backbone_feat, hmd_info)

    def _compute_hmd_from_pose(self, pose_3d: Tensor) -> Tensor:
        """Compute HMD info from 3D pose."""
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
        batch_data_samples: OptSampleList,
        recon_heatmaps: Optional[Tensor] = None
    ) -> Tuple[InstanceList, Tensor]:
        """Decode predictions."""

        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        pose_3d, recon_heatmaps_out, info = self.forward_lifting(
            backbone_feat, HMD_info.float()
        )

        if recon_heatmaps is None:
            recon_heatmaps = recon_heatmaps_out

        hmd_recons = self._compute_hmd_from_pose(pose_3d)

        # Decode 2D keypoints
        if recon_heatmaps is not None and self.decoder is not None:
            if self.decoder.support_batch_decoding:
                batch_keypoints, batch_scores = self.decoder.batch_decode(recon_heatmaps)
                if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                    batch_scores, batch_visibility = batch_scores
                else:
                    batch_visibility = [None] * len(batch_keypoints)
            else:
                batch_output_np = to_numpy(recon_heatmaps, unzip=True)
                batch_keypoints = []
                batch_scores = []
                batch_visibility = []
                for outputs in batch_output_np:
                    keypoints, scores = self.decoder.decode(outputs)
                    batch_keypoints.append(keypoints)
                    if isinstance(scores, tuple):
                        batch_scores.append(scores[0])
                        batch_visibility.append(scores[1])
                    else:
                        batch_scores.append(scores)
                        batch_visibility.append(None)
        else:
            B = pose_3d.shape[0]
            pose_2d = pose_3d[:, :, :2]
            batch_keypoints = pose_2d.detach().cpu().numpy() * 256
            batch_scores = [np.ones((16,)) for _ in range(B)]
            batch_visibility = [None] * B

        preds = []
        B = pose_3d.shape[0]
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

            pred = InstanceData(
                keypoints=kpts,
                keypoint_scores=scores,
                keypoint_3d=pose_3d[i:i+1].detach().cpu().numpy(),
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
            _, recon_heatmaps, _ = self.forward_lifting(backbone_feat, HMD_info.float())
            if recon_heatmaps is not None:
                pred_fields = [PixelData(heatmaps=hm) for hm in recon_heatmaps.detach()]
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

        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])

        pose_3d, recon_heatmaps, info = self.forward_lifting(backbone_feat, HMD_info.float())
        hmd_recon = self._compute_hmd_from_pose(pose_3d)

        losses = dict()

        # 3D Pose Losses
        pose_3d = pose_3d.reshape(-1, 16, 3)

        loss_pose_l2norm = self.loss_pose_l2norm_module(pose_3d, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pose_3d, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pose_3d, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # HMD Loss
        loss_hmd = self.loss_hmd_module(hmd_recon.double(), HMD_info.double())
        losses['loss_hmd'] = loss_hmd

        # Heatmap Reconstruction Loss
        if self.use_heatmap_recon and recon_heatmaps is not None:
            gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
            keypoint_weights = torch.cat([
                d.gt_instance_labels.keypoint_weights for d in batch_data_samples
            ])

            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmaps, gt_heatmaps, keypoint_weights
            )
            losses['loss_heatmap_recon'] = loss_heatmap_recon

            if train_cfg.get('compute_acc', True):
                _, avg_acc, _ = pose_pck_accuracy(
                    output=to_numpy(recon_heatmaps),
                    target=to_numpy(gt_heatmaps),
                    mask=to_numpy(keypoint_weights) > 0)
                losses['acc_pose'] = torch.tensor(avg_acc, device=gt_heatmaps.device)

        return losses
