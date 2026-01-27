# Copyright (c) OpenMMLab. All rights reserved.
"""
Hierarchical Pose Estimation Head (Option 2 v2)

Key Insight:
    - Stage 1: Upper Body 예측 (HMD Cross-Attention 활용)
    - Stage 2: Lower Body 예측 (Upper 정보 + 이미지 참조)
    - Cross-Attention으로 2D 위치 학습 (Self-Attention 중복 제거)
    - Heatmap Recon Loss로 2D supervision 보장

Architecture:
                    Backbone feat [2048, 8, 8]
                              │
                              ▼
                    Spatial Tokens [64, D]  ← 공유
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        Upper Branch                    Lower Branch
    ┌─────────────────┐           ┌─────────────────┐
    │ Cross-Attn (2D) │           │ Cross-Attn (2D) │
    │ Q: Upper Query  │           │ Q: Lower Query  │
    │ K/V: Spatial    │           │ K/V: Spatial    │
    │       │         │           │       │         │
    │       ▼         │           │       ▼         │
    │ Heatmap Recon   │           │ Cross-Attn      │
    │       │         │           │ (Upper ref)     │
    │       ▼         │           │       │         │
    │ HMD Cross-Attn  │──────────→│ Heatmap Recon   │
    │       │         │           │       │         │
    │       ▼         │           │       ▼         │
    │ Upper Pose[8,3] │           │ Lower Pose[8,3] │
    └─────────────────┘           └─────────────────┘
              │                           │
              └───────────┬───────────────┘
                          ▼
                  Full Pose [16, 3]

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


class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for spatial tokens."""

    def __init__(self, embed_dim: int, h: int = 8, w: int = 8):
        super().__init__()
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
    """Cross-Attention Layer with Pre-LayerNorm."""

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

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query: [B, N_q, D]
            key_value: [B, N_kv, D]
        Returns:
            output: [B, N_q, D]
        """
        # Cross-Attention
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        query = query + self.dropout(attn_out)

        # FFN
        query = query + self.ffn(self.norm_ffn(query))

        return query


class PerJointHeatmapDecoder(nn.Module):
    """Per-Joint Heatmap Decoder for reconstruction regularization."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_joints: int = 8,
        heatmap_size: int = 47,
        hidden_dim: int = 256
    ):
        super().__init__()
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
    """Cross-attention for HMD information."""

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
        attn_out, _ = self.cross_attn(joint_norm, hmd_tokens, hmd_tokens)
        return joint_tokens + self.dropout(attn_out)


class FusionLayer(nn.Module):
    """Fusion layer to combine Joint Tokens and Pose Embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pose_embed = nn.Linear(3, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, joint_tokens: Tensor, pose: Tensor) -> Tensor:
        """
        Args:
            joint_tokens: [B, 8, D] - 이미지 특징
            pose: [B, 8, 3] - 3D 좌표
        Returns:
            fused: [B, 8, D]
        """
        pose_emb = self.pose_embed(pose)
        fused = torch.cat([joint_tokens, pose_emb], dim=-1)
        fused = self.fusion(fused)
        return fused


class Pose3DHead(nn.Module):
    """3D Pose prediction head."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.head(tokens)


@MODELS.register_module()
class CustomEgoposeHierarchicalHead(BaseHead):
    """
    Hierarchical Pose Estimation Head.

    Stage 1: Upper Body (with HMD Cross-Attention)
    Stage 2: Lower Body (with Upper reference via Fusion)

    Key Features:
    - Shared Spatial Tokens (no duplication)
    - Cross-Attention for 2D learning (efficient)
    - Heatmap Recon Loss for 2D supervision
    - Hierarchical: Lower references Upper

    Args:
        in_channels: Input channels from backbone (2048)
        out_channels: Total number of keypoints (16)
        embed_dim: Embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_cross_layers: Number of cross-attention layers for 2D (default: 2)
        dropout: Dropout rate (default: 0.1)
        detach_upper: Whether to detach upper gradient to lower (default: False)
    """

    _version = 1

    UPPER_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7]
    LOWER_JOINTS = [8, 9, 10, 11, 12, 13, 14, 15]

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
        heatmap_size: int = 47,
        detach_upper: bool = False,
        # Loss configs
        loss_upper_heatmap_recon: ConfigType = dict(type='KeypointMSELoss', loss_weight=500),
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
        self.embed_dim = embed_dim
        self.heatmap_size = heatmap_size
        self.detach_upper = detach_upper

        # Build losses
        self.loss_upper_heatmap_recon = MODELS.build(loss_upper_heatmap_recon)
        self.loss_lower_heatmap_recon = MODELS.build(loss_lower_heatmap_recon)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # ===== Shared Components =====
        # Spatial projection (backbone → spatial tokens)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.spatial_pos_enc = PositionalEncoding2D(embed_dim, 8, 8)

        # ===== Stage 1: Upper Body =====
        # Learnable queries
        self.upper_queries = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)

        # Cross-attention layers (Query → Spatial)
        self.upper_cross_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])

        # Heatmap decoder
        self.upper_heatmap_decoder = PerJointHeatmapDecoder(
            embed_dim=embed_dim, num_joints=8, heatmap_size=heatmap_size
        )

        # HMD Cross-attention
        self.hmd_cross_attn = HMDCrossAttention(
            embed_dim=embed_dim, hmd_dim=9, num_heads=num_heads // 2, dropout=dropout
        )

        # 3D pose head
        self.upper_pose_head = Pose3DHead(embed_dim)

        # ===== Stage 2: Lower Body =====
        # Learnable queries
        self.lower_queries = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)

        # Cross-attention layers (Query → Spatial)
        self.lower_cross_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])

        # Cross-attention for Upper reference
        self.upper_ref_cross_attn = CrossAttentionLayer(embed_dim, num_heads, dropout)

        # Fusion layer
        self.fusion_layer = FusionLayer(embed_dim)

        # Heatmap decoder
        self.lower_heatmap_decoder = PerJointHeatmapDecoder(
            embed_dim=embed_dim, num_joints=8, heatmap_size=heatmap_size
        )

        # 3D pose head
        self.lower_pose_head = Pose3DHead(embed_dim)

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
        """Forward pass - returns None as we use forward_hierarchical."""
        return None

    def forward_hierarchical(
        self,
        backbone_feat: Tensor,
        hmd_info: Tensor
    ) -> dict:
        """
        Forward pass for hierarchical network.

        Args:
            backbone_feat: [B, 2048, 8, 8]
            hmd_info: [B, 9]

        Returns:
            dict with pose_3d, upper/lower outputs
        """
        B = backbone_feat.size(0)

        # ===== Shared: Spatial Tokens =====
        spatial_feat = self.spatial_proj(backbone_feat)
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, 64, D]
        spatial_tokens = self.spatial_pos_enc(spatial_tokens)

        # ===== Stage 1: Upper Body =====
        upper_tokens = self.upper_queries.expand(B, -1, -1).clone()

        # Cross-attention: Upper Queries → Spatial Tokens (2D learning)
        for layer in self.upper_cross_layers:
            upper_tokens = layer(upper_tokens, spatial_tokens)

        # Heatmap reconstruction (for 2D supervision)
        upper_heatmaps = self.upper_heatmap_decoder(upper_tokens)

        # HMD Cross-attention (depth reference)
        upper_tokens_with_hmd = self.hmd_cross_attn(upper_tokens, hmd_info)

        # 3D pose prediction
        upper_pose = self.upper_pose_head(upper_tokens_with_hmd)  # [B, 8, 3]

        # ===== Fusion: Upper → Lower =====
        if self.detach_upper:
            upper_tokens_detached = upper_tokens.detach()
            upper_pose_detached = upper_pose.detach()
        else:
            upper_tokens_detached = upper_tokens
            upper_pose_detached = upper_pose

        fused_upper = self.fusion_layer(upper_tokens_detached, upper_pose_detached)

        # ===== Stage 2: Lower Body =====
        lower_tokens = self.lower_queries.expand(B, -1, -1).clone()

        # Cross-attention: Lower Queries → Spatial Tokens (2D learning)
        for layer in self.lower_cross_layers:
            lower_tokens = layer(lower_tokens, spatial_tokens)

        # Cross-attention: Lower Tokens → Fused Upper (hierarchical reference)
        lower_tokens = self.upper_ref_cross_attn(lower_tokens, fused_upper)

        # Heatmap reconstruction (for 2D supervision)
        lower_heatmaps = self.lower_heatmap_decoder(lower_tokens)

        # 3D pose prediction
        lower_pose = self.lower_pose_head(lower_tokens)  # [B, 8, 3]

        # ===== Combine =====
        pose_3d = torch.cat([upper_pose, lower_pose], dim=1)  # [B, 16, 3]

        return {
            'pose_3d': pose_3d,
            'upper_pose': upper_pose,
            'lower_pose': lower_pose,
            'upper_tokens': upper_tokens,
            'lower_tokens': lower_tokens,
            'upper_heatmaps': upper_heatmaps,
            'lower_heatmaps': lower_heatmaps,
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

        outputs = self.forward_hierarchical(backbone_feat, HMD_info.float())
        pose_3d = outputs['pose_3d']

        hmd_recons = self._compute_hmd_from_pose(pose_3d)

        # Combine heatmaps
        upper_heatmaps = outputs['upper_heatmaps']
        lower_heatmaps = outputs['lower_heatmaps']
        full_heatmaps = torch.cat([upper_heatmaps, lower_heatmaps], dim=1)

        # Decode 2D keypoints
        B = pose_3d.shape[0]
        if self.decoder is not None:
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

            gen_heatmap = full_heatmaps[i:i+1].detach().cpu().numpy()

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
            outputs = self.forward_hierarchical(backbone_feat, HMD_info.float())
            full_heatmaps = torch.cat([
                outputs['upper_heatmaps'], outputs['lower_heatmaps']
            ], dim=1)
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
        outputs = self.forward_hierarchical(backbone_feat, HMD_info.float())
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
        gt_upper_heatmaps = gt_heatmaps[:, self.UPPER_JOINTS, :, :]
        upper_keypoint_weights = keypoint_weights[:, self.UPPER_JOINTS]

        loss_upper_hm = self.loss_upper_heatmap_recon(
            outputs['upper_heatmaps'], gt_upper_heatmaps, upper_keypoint_weights
        )
        losses['loss_upper_hm_recon'] = loss_upper_hm

        # ===== Lower Body Heatmap Reconstruction Loss =====
        gt_lower_heatmaps = gt_heatmaps[:, self.LOWER_JOINTS, :, :]
        lower_keypoint_weights = keypoint_weights[:, self.LOWER_JOINTS]

        loss_lower_hm = self.loss_lower_heatmap_recon(
            outputs['lower_heatmaps'], gt_lower_heatmaps, lower_keypoint_weights
        )
        losses['loss_lower_hm_recon'] = loss_lower_hm

        # ===== Accuracy =====
        if train_cfg.get('compute_acc', True):
            # Upper body accuracy
            _, upper_acc, _ = pose_pck_accuracy(
                output=to_numpy(outputs['upper_heatmaps']),
                target=to_numpy(gt_upper_heatmaps),
                mask=to_numpy(upper_keypoint_weights) > 0
            )
            losses['acc_upper'] = torch.tensor(upper_acc, device=gt_heatmaps.device)

            # Lower body accuracy
            _, lower_acc, _ = pose_pck_accuracy(
                output=to_numpy(outputs['lower_heatmaps']),
                target=to_numpy(gt_lower_heatmaps),
                mask=to_numpy(lower_keypoint_weights) > 0
            )
            losses['acc_lower'] = torch.tensor(lower_acc, device=gt_heatmaps.device)

            # Full body accuracy
            losses['acc_pose'] = (losses['acc_upper'] + losses['acc_lower']) / 2

        return losses
