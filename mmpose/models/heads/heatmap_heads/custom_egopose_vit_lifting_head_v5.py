# Copyright (c) OpenMMLab. All rights reserved.
"""
ViT-Style Attention Lifting Head v5 - Hybrid Attention

Key Changes from v4:
- Self-Attention [80×80] → Cross-Attention [16×64] + Self-Attention [16×16]
- 역할 분리: Cross(위치 찾기) + Self(skeleton 관계)
- 계산 효율: 6400 → 1280 attention weights (5배 감소)
- Gradient Scale: Heatmap gradient를 줄여 3D 학습에 집중

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Backbone feat [2048, 8, 8]                                 │
    │         ↓                                                   │
    │  Spatial Tokens [64, D]                                     │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  Stage 1: Cross-Attention (J → S)       │                │
    │  │  Q: Joint Queries [16, D]               │                │
    │  │  K/V: Spatial Tokens [64, D]            │                │
    │  │  → 각 관절이 이미지에서 위치 찾기         │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  Stage 2: Self-Attention (J → J) × N    │                │
    │  │  Q=K=V: Joint Tokens [16, D]            │                │
    │  │  → 관절 간 skeleton 관계 학습            │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  Joint Tokens [16, D]                                       │
    │         │                                                   │
    │         ├── Heatmap Decoder (scaled gradient)               │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  Stage 3: HMD Cross-Attention           │                │
    │  │  Q: Joint Tokens, K/V: HMD Tokens       │                │
    │  │  → 3D depth reference 추가               │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │    3D Pose Head → [16, 3]                                   │
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


class GradientScale(torch.autograd.Function):
    """Scale gradients during backward pass."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def gradient_scale(x, scale):
    """Apply gradient scaling."""
    return GradientScale.apply(x, scale)


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


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer (J → S).

    Joint queries attend to spatial tokens to find their locations.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, joint_tokens: Tensor, spatial_tokens: Tensor) -> Tensor:
        """
        Args:
            joint_tokens: [B, 16, D] - queries
            spatial_tokens: [B, 64, D] - keys/values
        Returns:
            updated_joints: [B, 16, D]
        """
        # Cross-attention (Pre-LayerNorm)
        q = self.norm_q(joint_tokens)
        kv = self.norm_kv(spatial_tokens)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        joint_tokens = joint_tokens + self.dropout(attn_out)

        # FFN
        joint_tokens = joint_tokens + self.ffn(self.norm_ffn(joint_tokens))

        return joint_tokens


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention Layer (J → J).

    Joint tokens attend to each other for skeleton relationships.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, joint_tokens: Tensor) -> Tensor:
        """
        Args:
            joint_tokens: [B, 16, D]
        Returns:
            refined_joints: [B, 16, D]
        """
        # Self-attention
        x_norm = self.norm1(joint_tokens)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        joint_tokens = joint_tokens + self.dropout1(attn_out)

        # FFN
        joint_tokens = joint_tokens + self.ffn(self.norm2(joint_tokens))

        return joint_tokens


class HMDCrossAttention(nn.Module):
    """
    Cross-attention: Joint tokens query HMD tokens for 3D depth reference.

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

        # HMD → 3 tokens [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention
        joint_norm = self.norm(joint_tokens)
        attn_out, _ = self.cross_attn(
            query=joint_norm,
            key=hmd_tokens,
            value=hmd_tokens
        )

        return joint_tokens + self.dropout(attn_out)


class PerJointHeatmapDecoder(nn.Module):
    """Per-Joint Heatmap Decoder for reconstruction regularization."""

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


class HybridAttentionLiftingNetwork(nn.Module):
    """
    Hybrid Attention Lifting Network (v5).

    Key improvements:
    1. Cross-Attention (J→S): 관절이 spatial에서 위치 찾기 [16×64]
    2. Self-Attention (J→J): 관절 간 skeleton 관계 [16×16]
    3. Gradient Scaling: Heatmap gradient 감소

    Pipeline:
    1. Backbone feat → Spatial tokens [64, D]
    2. Cross-Attention: Joint queries → Spatial tokens
    3. Self-Attention: Joint ↔ Joint (skeleton)
    4. Heatmap Reconstruction (scaled gradient)
    5. HMD Cross-Attention (3D depth)
    6. 3D Pose Head
    """

    def __init__(
        self,
        num_joints: int = 16,
        embed_dim: int = 256,
        backbone_channels: int = 2048,
        spatial_size: int = 8,
        heatmap_size: int = 47,
        hmd_dim: int = 9,
        num_heads: int = 8,
        num_cross_layers: int = 1,
        num_self_layers: int = 2,
        dropout: float = 0.1,
        use_hmd: bool = True,
        use_heatmap_recon: bool = True,
        heatmap_grad_scale: float = 0.1
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.use_hmd = use_hmd
        self.use_heatmap_recon = use_heatmap_recon
        self.heatmap_grad_scale = heatmap_grad_scale

        # Spatial projection
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # Positional encoding
        self.spatial_pos_enc = PositionalEncoding2D(embed_dim, spatial_size, spatial_size)

        # Learnable Joint Queries
        self.joint_queries = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # Joint positional embedding (skeleton structure hint)
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # Stage 1: Cross-Attention (J → S)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])

        # Stage 2: Self-Attention (J → J)
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_self_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Heatmap Decoder (for regularization)
        if use_heatmap_recon:
            self.heatmap_decoder = PerJointHeatmapDecoder(
                embed_dim=embed_dim,
                num_joints=num_joints,
                heatmap_size=heatmap_size
            )

        # Stage 3: HMD Cross-Attention
        if use_hmd:
            self.hmd_cross_attn = HMDCrossAttention(
                embed_dim=embed_dim,
                hmd_dim=hmd_dim,
                num_heads=num_heads // 2,
                dropout=dropout
            )

        # 3D Pose Head
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
        nn.init.normal_(self.joint_queries, std=0.02)
        nn.init.normal_(self.joint_pos_embed, std=0.02)

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

        # 1. Spatial tokens [B, 64, D]
        spatial_feat = self.spatial_proj(backbone_feat)
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)
        spatial_tokens = self.spatial_pos_enc(spatial_tokens)

        # 2. Joint queries [B, 16, D]
        joint_tokens = self.joint_queries.expand(B, -1, -1).clone()
        joint_tokens = joint_tokens + self.joint_pos_embed

        # Stage 1: Cross-Attention (J → S)
        for cross_layer in self.cross_attn_layers:
            joint_tokens = cross_layer(joint_tokens, spatial_tokens)

        # Stage 2: Self-Attention (J → J)
        for self_layer in self.self_attn_layers:
            joint_tokens = self_layer(joint_tokens)

        joint_tokens = self.norm(joint_tokens)

        # Heatmap Reconstruction (with gradient scaling)
        recon_heatmaps = None
        if self.use_heatmap_recon:
            # Scale gradient: forward는 정상, backward는 0.1배
            scaled_tokens = gradient_scale(joint_tokens, self.heatmap_grad_scale)
            recon_heatmaps = self.heatmap_decoder(scaled_tokens)

        # Stage 3: HMD Cross-Attention
        if self.use_hmd and hmd_info is not None:
            joint_tokens = self.hmd_cross_attn(joint_tokens, hmd_info)

        # 3D Pose prediction
        pose_3d = self.head_3d(joint_tokens)

        info = {
            'spatial_tokens': spatial_tokens,
            'joint_tokens': joint_tokens
        }

        return pose_3d, recon_heatmaps, info


@MODELS.register_module()
class CustomEgoposeViTLiftingHeadV5(BaseHead):
    """
    ViT-Style Lifting Head v5 - Hybrid Attention.

    Key improvements over v4:
    1. Hybrid Attention: Cross(J→S) + Self(J→J) instead of Self([S+J]→[S+J])
    2. Gradient Scaling: Reduce heatmap gradient influence on joint tokens
    3. Computational Efficiency: 6400 → 1280 attention weights

    Args:
        in_channels: Input channels from backbone (2048)
        out_channels: Number of keypoints (16)
        embed_dim: Transformer embedding dimension (256)
        num_heads: Number of attention heads (8)
        num_cross_layers: Number of cross-attention layers (1)
        num_self_layers: Number of self-attention layers (2)
        heatmap_grad_scale: Gradient scale for heatmap loss (0.1)
    """

    _version = 5

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_cross_layers: int = 1,
        num_self_layers: int = 2,
        dropout: float = 0.1,
        heatmap_size: int = 47,
        use_hmd: bool = True,
        use_heatmap_recon: bool = True,
        heatmap_grad_scale: float = 0.1,
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

        # Hybrid Attention Lifting Network
        self.lifting_network = HybridAttentionLiftingNetwork(
            num_joints=out_channels,
            embed_dim=embed_dim,
            backbone_channels=in_channels,
            spatial_size=8,
            heatmap_size=heatmap_size,
            hmd_dim=9,
            num_heads=num_heads,
            num_cross_layers=num_cross_layers,
            num_self_layers=num_self_layers,
            dropout=dropout,
            use_hmd=use_hmd,
            use_heatmap_recon=use_heatmap_recon,
            heatmap_grad_scale=heatmap_grad_scale
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
        pose_3d = pose_3d.view(-1, 16, 3)

        loss_pose_l2norm = self.loss_pose_l2norm_module(pose_3d, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pose_3d, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pose_3d, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # HMD Loss
        loss_hmd = self.loss_hmd_module(
            hmd_recon.double(), HMD_info.double()
        )
        losses['loss_hmd'] = loss_hmd

        # Heatmap Reconstruction Loss (gradient already scaled in forward)
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
