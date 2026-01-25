# Copyright (c) OpenMMLab. All rights reserved.
"""
ViT-Style Attention Lifting Head (v2 - Reconstruction Regularized)

Key Insight from EfficientHeatmapDecoder:
    Heatmap reconstruction is NOT for decoding heatmaps,
    but for INJECTING JOINT INFORMATION into the latent representation.
    Reconstruction loss forces joint tokens to contain 2D spatial info.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Backbone feat [2048, 8, 8]                                 │
    │         ↓                                                   │
    │  Spatial Tokens [64, D] + Learnable Joint Queries [16, D]  │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │     Self-Attention (4 layers)           │                │
    │  │     Spatial ↔ Joint bidirectional       │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  Joint Tokens [16, D]                                       │
    │         │                                                   │
    │         ├───────────────────────┐                           │
    │         │                       ↓                           │
    │         │              ┌────────────────────┐               │
    │         │              │  Heatmap Decoder   │               │
    │         │              │  (Reconstruction)  │               │
    │         │              └────────────────────┘               │
    │         │                       ↓                           │
    │         │              Recon Heatmap [16,47,47]             │
    │         │              (loss injects 2D info)               │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │     HMD Cross-Attention                 │                │
    │  │     (adds 3D depth reference)           │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  ┌────────────────┐                                         │
    │  │  3D Pose Head  │                                         │
    │  │  Linear(D, 3)  │                                         │
    │  └────────────────┘                                         │
    │         ↓                                                   │
    │    3D Pose [16, 3]                                          │
    └─────────────────────────────────────────────────────────────┘

Information Flow:
    1. Self-Attention: Spatial ↔ Joint bidirectional interaction
    2. Heatmap Reconstruction: Forces 2D joint localization into tokens
    3. HMD Cross-Attention: Adds 3D depth reference from HMD
    4. 3D Pose Head: Final 3D pose from enriched tokens

References:
- ViTPose (NeurIPS 2022): Simple Vision Transformer Baselines
- TokenPose (ICCV 2021): Learning Keypoint Tokens
- xRegopose: Encoder-Decoder with reconstruction loss
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


class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for spatial tokens."""

    def __init__(self, embed_dim: int, h: int = 8, w: int = 8):
        super().__init__()
        self.embed_dim = embed_dim

        # Create 2D positional encoding
        pe = torch.zeros(h * w, embed_dim)

        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w).flatten()
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1).flatten()

        div_term = torch.exp(torch.arange(0, embed_dim // 2, 2) *
                            -(math.log(10000.0) / (embed_dim // 2)))

        pe[:, 0::4] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 1::4] = torch.cos(x_pos.unsqueeze(1) * div_term)
        pe[:, 2::4] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, 3::4] = torch.cos(y_pos.unsqueeze(1) * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, H*W, D]

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to spatial tokens."""
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
        # Pre-LayerNorm Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # Pre-LayerNorm MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PerJointHeatmapDecoder(nn.Module):
    """
    Per-Joint Heatmap Decoder for reconstruction regularization.

    Takes each joint token [D] and decodes it to a single-channel heatmap [H, W].
    This forces the joint tokens to contain 2D spatial localization information.

    Architecture (per joint):
        Joint Token [D] → MLP → Spatial Feature [D', 1, 1]
                              → Upsample → Heatmap [1, H, W]

    Total output: [B, num_joints, H, W]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_joints: int = 16,
        heatmap_size: int = 47,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # Shared MLP for all joints: [D] → [hidden_dim]
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Progressive upsampling: 1x1 → 3x3 → 6x6 → 12x12 → 24x24 → 47x47
        self.upsample = nn.Sequential(
            # 1x1 → 3x3
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # 3x3 → 6x6
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # 6x6 → 12x12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # 12x12 → 24x24
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # 24x24 → 47x47 (bilinear + conv)
            nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, joint_tokens: Tensor) -> Tensor:
        """
        Args:
            joint_tokens: [B, num_joints, D]

        Returns:
            heatmaps: [B, num_joints, H, W]
        """
        B, N, D = joint_tokens.shape  # B, 16, 256

        # Process all joints together
        tokens_flat = joint_tokens.view(B * N, D)  # [B*16, D]
        features = self.fc(tokens_flat)  # [B*16, hidden_dim]

        # Reshape for conv: [B*16, hidden_dim, 1, 1]
        features = features.view(B * N, -1, 1, 1)

        # Upsample to heatmap: [B*16, 1, H, W]
        heatmaps = self.upsample(features)

        # Reshape: [B, 16, H, W]
        heatmaps = heatmaps.view(B, N, self.heatmap_size, self.heatmap_size)

        return heatmaps


class HMDCrossAttention(nn.Module):
    """
    Cross-attention: Joint tokens query HMD tokens for 3D reference.

    Q: Joint tokens [B, 16, D]
    K/V: HMD tokens [B, 3, D] - (head, right_hand, left_hand)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hmd_dim: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # HMD → 3 tokens
        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 3)  # 3 tokens
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
        """
        Args:
            joint_tokens: [B, 16, D]
            hmd_info: [B, 9]
        Returns:
            refined_joints: [B, 16, D]
        """
        B = joint_tokens.size(0)

        # HMD → 3 tokens [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention (Pre-LayerNorm)
        joint_norm = self.norm(joint_tokens)
        attn_out, _ = self.cross_attn(
            query=joint_norm,
            key=hmd_tokens,
            value=hmd_tokens
        )

        return joint_tokens + self.dropout(attn_out)


class ViTLiftingNetwork(nn.Module):
    """
    ViT-Style Lifting Network (v2 - Reconstruction Regularized).

    Uses learnable joint queries and self-attention, with heatmap
    reconstruction to inject 2D joint information into joint tokens.

    Pipeline:
    1. Backbone feat → Spatial tokens
    2. Concat [Spatial; Joint queries]
    3. Self-Attention (bidirectional)
    4. Extract joint tokens
    5. Heatmap Reconstruction (forces 2D info into tokens)
    6. HMD Cross-Attention (adds 3D depth reference)
    7. 3D Pose Head
    """

    def __init__(
        self,
        num_joints: int = 16,
        embed_dim: int = 256,
        backbone_channels: int = 2048,
        spatial_size: int = 8,  # 8x8 spatial grid
        heatmap_size: int = 47,
        hmd_dim: int = 9,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_hmd: bool = True,
        use_heatmap_recon: bool = True
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.spatial_tokens_count = spatial_size * spatial_size  # 64
        self.use_hmd = use_hmd
        self.use_heatmap_recon = use_heatmap_recon

        # 1. Backbone feature → Spatial tokens
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # 2. Positional encoding for spatial tokens
        self.spatial_pos_enc = PositionalEncoding2D(embed_dim, spatial_size, spatial_size)

        # 3. Learnable Joint Queries (key innovation!)
        self.joint_queries = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # 4. Learnable type embedding (spatial vs joint)
        self.spatial_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 5. Transformer Encoder (Self-Attention)
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

        # 6. Heatmap Reconstruction Decoder (forces 2D info into joint tokens)
        if use_heatmap_recon:
            self.heatmap_decoder = PerJointHeatmapDecoder(
                embed_dim=embed_dim,
                num_joints=num_joints,
                heatmap_size=heatmap_size,
                hidden_dim=256
            )

        # 7. HMD Cross-Attention (adds 3D depth reference)
        if use_hmd:
            self.hmd_cross_attn = HMDCrossAttention(
                embed_dim=embed_dim,
                hmd_dim=hmd_dim,
                num_heads=num_heads // 2,
                dropout=dropout
            )

        # 8. 3D Pose Head (direct 3D prediction)
        self.head_3d = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize joint queries
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
            backbone_feat: [B, 2048, 8, 8] backbone features
            hmd_info: [B, 9] HMD information (optional)

        Returns:
            pose_3d: [B, 16, 3] predicted 3D pose
            recon_heatmaps: [B, 16, 47, 47] reconstructed heatmaps (or None)
            info: dict with intermediate outputs
        """
        B = backbone_feat.size(0)

        # 1. Backbone → Spatial tokens [B, 64, D]
        spatial_feat = self.spatial_proj(backbone_feat)  # [B, D, 8, 8]
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, 64, D]

        # Add positional encoding
        spatial_tokens = self.spatial_pos_enc(spatial_tokens)

        # Add type embedding
        spatial_tokens = spatial_tokens + self.spatial_type_embed

        # 2. Expand joint queries for batch [B, 16, D]
        joint_tokens = self.joint_queries.expand(B, -1, -1).clone()
        joint_tokens = joint_tokens + self.joint_type_embed

        # 3. Concatenate [spatial; joint] = [B, 80, D]
        tokens = torch.cat([spatial_tokens, joint_tokens], dim=1)

        # 4. Self-Attention layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)

        # 5. Extract joint tokens (last 16)
        joint_tokens = tokens[:, -self.num_joints:, :]  # [B, 16, D]

        # 6. Heatmap Reconstruction (BEFORE HMD, forces 2D info)
        recon_heatmaps = None
        if self.use_heatmap_recon:
            recon_heatmaps = self.heatmap_decoder(joint_tokens)  # [B, 16, 47, 47]

        # 7. HMD Cross-Attention (adds 3D depth reference)
        if self.use_hmd and hmd_info is not None:
            joint_tokens = self.hmd_cross_attn(joint_tokens, hmd_info)

        # 8. 3D Pose prediction
        pose_3d = self.head_3d(joint_tokens)  # [B, 16, 3]

        info = {
            'spatial_tokens': spatial_tokens,
            'joint_tokens_before_hmd': tokens[:, -self.num_joints:, :],
            'joint_tokens_after_hmd': joint_tokens
        }

        return pose_3d, recon_heatmaps, info


@MODELS.register_module()
class CustomEgoposeViTLiftingHead(BaseHead):
    """
    ViT-Style Lifting Head (v2 - Reconstruction Regularized).

    Key Features:
    - Learnable Joint Queries: No soft_argmax bottleneck
    - Self-Attention: Bidirectional spatial ↔ joint interaction
    - Heatmap Reconstruction: Forces 2D joint info into joint tokens
    - HMD Cross-Attention: Adds 3D depth reference
    - Direct 3D Prediction: Single head for (x, y, z)

    Information Flow:
    1. Self-Attention → Joint tokens contain spatial info
    2. Heatmap Reconstruction Loss → Forces 2D localization into tokens
    3. HMD Cross-Attention → Adds 3D depth reference
    4. 3D Pose Head → Final 3D pose

    Args:
        in_channels: Input channels from backbone (2048 for ResNet-101)
        out_channels: Number of keypoints (16)
        embed_dim: Transformer embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        use_hmd: Whether to use HMD cross-attention (default: True)
        use_heatmap_recon: Whether to use heatmap reconstruction loss (default: True)
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        # ViT Lifting params
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        heatmap_size: int = 47,
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
        # Legacy params (ignored, for config compatibility)
        loss: OptConfigType = None,
        loss_2d: OptConfigType = None,
        use_heatmap_loss: bool = True,
        deconv_out_channels: OptIntSeq = None,
        deconv_kernel_sizes: OptIntSeq = None,
        deconv_stride_sizes: OptIntSeq = None,
        final_layer: dict = None,
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

        # ViT Lifting Network (v2)
        self.lifting_network = ViTLiftingNetwork(
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
            use_hmd=use_hmd,
            use_heatmap_recon=use_heatmap_recon
        )

        # Decoder (for 2D keypoint decoding from reconstructed heatmaps)
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
        """Forward pass - returns None as heatmaps come from reconstruction."""
        return None

    def forward_lifting(
        self,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], dict]:
        """
        Forward pass for ViT-style lifting.

        Args:
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]

        Returns:
            pose_3d: [B, 16, 3]
            recon_heatmaps: [B, 16, 47, 47] or None
            info: dict
        """
        return self.lifting_network(backbone_feat, hmd_info)

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
        batch_data_samples: OptSampleList,
        recon_heatmaps: Optional[Tensor] = None
    ) -> Tuple[InstanceList, Tensor]:
        """Decode predictions."""

        # Get HMD info
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # ViT Lifting
        pose_3d, recon_heatmaps_out, info = self.forward_lifting(
            backbone_feat, HMD_info.float()
        )

        # Use passed heatmaps or the ones from forward_lifting
        if recon_heatmaps is None:
            recon_heatmaps = recon_heatmaps_out

        # HMD reconstruction
        hmd_recons = self._compute_hmd_from_pose(pose_3d)

        # Decode 2D keypoints from reconstructed heatmaps if available
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
            # Use pose_3d[:, :, :2] as 2D keypoints
            B = pose_3d.shape[0]
            pose_2d = pose_3d[:, :, :2]  # [B, 16, 2]
            batch_keypoints = pose_2d.detach().cpu().numpy() * 256  # Scale to input size
            batch_scores = [np.ones((16,)) for _ in range(B)]
            batch_visibility = [None] * B

        preds = []
        B = pose_3d.shape[0]
        for i in range(B):
            if isinstance(batch_keypoints, np.ndarray):
                kpts = batch_keypoints[i:i+1]
            else:
                kpts = batch_keypoints[i]

            if isinstance(batch_scores, np.ndarray):
                scores = batch_scores[i]
            else:
                scores = batch_scores[i]

            pred = InstanceData(
                keypoints=kpts,
                keypoint_scores=scores,
                keypoint_3d=pose_3d[i:i+1],
                hmd_recon=hmd_recons[i:i+1]
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
            # Get reconstructed heatmaps
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

        # Get HMD info and GT
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])

        # ViT Lifting forward (returns pose_3d, recon_heatmaps, info)
        pose_3d, recon_heatmaps, info = self.forward_lifting(backbone_feat, HMD_info.float())

        # HMD reconstruction from predicted pose
        hmd_recon = self._compute_hmd_from_pose(pose_3d)

        losses = dict()

        # ===== 3D Pose Losses =====
        pose_3d = pose_3d.view(-1, 16, 3)

        loss_pose_l2norm = self.loss_pose_l2norm_module(pose_3d, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pose_3d, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pose_3d, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # ===== HMD Loss =====
        loss_hmd = self.loss_hmd_module(
            hmd_recon.double(), HMD_info.double()
        )
        losses['loss_hmd'] = loss_hmd

        # ===== Heatmap Reconstruction Loss (key for injecting 2D info) =====
        if self.use_heatmap_recon and recon_heatmaps is not None:
            gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
            keypoint_weights = torch.cat([
                d.gt_instance_labels.keypoint_weights for d in batch_data_samples
            ])

            # Reconstruction loss forces joint tokens to encode 2D joint locations
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmaps, gt_heatmaps, keypoint_weights
            )
            losses['loss_heatmap_recon'] = loss_heatmap_recon

            # Accuracy from reconstructed heatmap
            if train_cfg.get('compute_acc', True):
                _, avg_acc, _ = pose_pck_accuracy(
                    output=to_numpy(recon_heatmaps),
                    target=to_numpy(gt_heatmaps),
                    mask=to_numpy(keypoint_weights) > 0)
                losses['acc_pose'] = torch.tensor(avg_acc, device=gt_heatmaps.device)

        return losses
