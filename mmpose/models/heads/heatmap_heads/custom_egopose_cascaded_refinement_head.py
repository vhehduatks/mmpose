# Copyright (c) OpenMMLab. All rights reserved.
"""
Cascaded Pose Refinement Head with Kinematic Prior

Two-stage architecture:
  Stage 1: Standard Baseline (Deconv → Heatmap → Encoder(GAP+HMD) → Z → PoseDecoder → Coarse 3D)
  Stage 2: Kinematic-Aware Refinement (Grid Sample + Kinematic Features + Shared MLP → Δpose)

Key insight from 20 experiments: The Baseline's unified MLP pipeline is the only
architecture that works for lower body. Instead of replacing it, this adds a
lightweight refinement stage that provides per-joint spatial features and explicit
kinematic structure using the same stable MLP building blocks.

Stage 2 components:
  1. Per-joint spatial features via grid_sample at predicted 2D locations
  2. Pose context encoding (full 48-dim coarse pose → 128)
  3. Kinematic chain features (15 bone vectors + lengths → 64)
  4. Shared RefinementMLP with BatchNorm + Dropout + residual → Δpose [16, 3]
  5. Refined Pose = Coarse Pose + Δpose

Parameters: Stage 2 adds ~400K params (<1% of total ~61M).
"""
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead

from mmengine.structures import InstanceData

import numpy as np
import math
from .blocks import PoseDecoder, HeatmapDecoder, EfficientHeatmapDecoder

OptIntSeq = Optional[Sequence[int]]

import torch.nn as nn


# XR-EgoPose skeleton: (parent_idx, child_idx) pairs for 15 limbs
EGOPOSE_SKELETON = [
    (0, 1),   # Spine2 -> Head
    (0, 2),   # Spine2 -> LeftArm
    (2, 3),   # LeftArm -> LeftForeArm
    (3, 4),   # LeftForeArm -> LeftHand
    (0, 5),   # Spine2 -> RightArm
    (5, 6),   # RightArm -> RightForeArm
    (6, 7),   # RightForeArm -> RightHand
    (0, 8),   # Spine2 -> LeftUpLeg
    (8, 9),   # LeftUpLeg -> LeftLeg
    (9, 10),  # LeftLeg -> LeftFoot
    (10, 11), # LeftFoot -> LeftToeBase
    (0, 12),  # Spine2 -> RightUpLeg
    (12, 13), # RightUpLeg -> RightLeg
    (13, 14), # RightLeg -> RightFoot
    (14, 15), # RightFoot -> RightToeBase
]


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


def soft_argmax_2d(heatmaps, temperature=1.0):
    """Differentiable 2D coordinate extraction from heatmaps.

    Args:
        heatmaps: [B, K, H, W] predicted heatmaps
        temperature: softmax temperature

    Returns:
        coords: [B, K, 2] normalized coordinates in [0, 1]
        confidence: [B, K] max heatmap values
    """
    B, K, H, W = heatmaps.shape
    hm_flat = heatmaps.reshape(B, K, -1)
    probs = F.softmax(hm_flat / temperature, dim=-1)

    y_coords = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
    x_coords = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    y_flat = y_grid.reshape(-1)
    x_flat = x_grid.reshape(-1)

    exp_y = (probs * y_flat).sum(dim=-1)  # [B, K]
    exp_x = (probs * x_flat).sum(dim=-1)

    coords = torch.stack([
        exp_x / max(W - 1, 1),
        exp_y / max(H - 1, 1),
    ], dim=-1)  # [B, K, 2]

    confidence = hm_flat.max(dim=-1)[0]  # [B, K]

    return coords, confidence


def compute_kinematic_features(pose_3d):
    """Compute bone direction vectors and lengths from 3D pose.

    Args:
        pose_3d: [B, 16, 3]

    Returns:
        kin_feat: [B, 60] (15 bones × (3 direction + 1 length))
    """
    kin_feats = []
    for parent, child in EGOPOSE_SKELETON:
        bone_vec = pose_3d[:, child] - pose_3d[:, parent]  # [B, 3]
        bone_len = torch.norm(bone_vec, dim=-1, keepdim=True)  # [B, 1]
        kin_feats.append(torch.cat([bone_vec, bone_len], dim=-1))  # [B, 4]
    return torch.cat(kin_feats, dim=-1)  # [B, 60]


def preprocess_hmd_data_batch(p3d):
    """Compute HMD reconstruction from predicted 3D pose."""
    if not isinstance(p3d, torch.Tensor):
        p3d = torch.tensor(p3d, dtype=torch.float32)

    head = p3d[:, 0]
    right_hand = p3d[:, 7]
    left_hand = p3d[:, 4]

    midpoint = (right_hand + left_hand) / 2
    z_axis = midpoint - head
    z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-8)

    hand_vector = right_hand - left_hand
    x_axis = torch.cross(z_axis, hand_vector, dim=1)
    x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-8)

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

    preprocessed_hmd = torch.cat([
        right_local, left_local,
        hand_distance.unsqueeze(1),
        right_distance.unsqueeze(1),
        left_distance.unsqueeze(1)
    ], dim=1)

    return preprocessed_hmd


# =========================================================================
# Stage 1 components (same as Baseline)
# =========================================================================

class Encoder(nn.Module):
    def __init__(self, num_classes=16, output_size=64, hmd_info_size=9):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.Linear(hmd_info_size, 36)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(256 + 36, output_size)
        self.linear2_ = nn.Linear(256, output_size)
        self.lrelu5 = nn.LeakyReLU(0.2)

    def forward(self, hm, hmd=None):
        hm = self.conv1(hm)
        hm = self.lrelu1(hm)
        hm = self.conv2(hm)
        hm = self.lrelu2(hm)
        hm = self.conv3(hm)
        hm = self.lrelu3(hm)

        hm_avgpool = self.avr_pool(hm).reshape(-1, 256)

        if hmd is not None:
            hmd = self.linear1(hmd)
            hmd = self.lrelu4(hmd)
            x = torch.cat((hm_avgpool, hmd), dim=1).to(torch.float32)
            x = self.linear2(x)
        else:
            x = self.linear2_(hm_avgpool)

        x = self.lrelu5(x)
        return x


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        return x + y


class LinearModel(nn.Module):
    def __init__(self, input_size=20, num_classes=16,
                 linear_size=512, num_stage=1, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = input_size
        self.output_size = num_classes * 3

        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = nn.ModuleList([
            Linear(self.linear_size, self.p_dropout) for _ in range(num_stage)
        ])

        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.linear_stages:
            y = stage(y)
        y = self.w2(y)
        y = y.reshape(-1, self.output_size // 3, 3)
        return y


# =========================================================================
# Stage 2 components (new)
# =========================================================================

class RefinementBlock(nn.Module):
    """Residual block with BatchNorm + ReLU + Dropout (same style as Baseline)."""

    def __init__(self, size, p_dropout=0.5):
        super(RefinementBlock, self).__init__()
        self.w1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.w2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)
        return x + y


class RefinementMLP(nn.Module):
    """Shared per-joint refinement network.

    Takes per-joint features and predicts Δpose [3] for residual correction.
    Input is processed as [B*K, input_size] to share weights across joints.
    """

    def __init__(self, input_size=323, hidden_size=256,
                 num_stage=1, p_dropout=0.5):
        super(RefinementMLP, self).__init__()
        self.w1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.stages = nn.ModuleList([
            RefinementBlock(hidden_size, p_dropout)
            for _ in range(num_stage)
        ])

        self.w_out = nn.Linear(hidden_size, 3)

    def forward(self, x):
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.stages:
            y = stage(y)
        return self.w_out(y)


# =========================================================================
# Main Head
# =========================================================================

@MODELS.register_module()
class CustomEgoposeCascadedRefinementHead(BaseHead):
    """Cascaded Pose Refinement Head.

    Stage 1: Standard Baseline (Deconv→Heatmap→Encoder→Z→PoseDecoder→Coarse3D)
    Stage 2: Kinematic-Aware Refinement (GridSample + KinFeatures + MLP → Δpose)
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4),
                 deconv_stride_sizes: OptIntSeq = (2, 2),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 # Stage 1 losses
                 loss: ConfigType = dict(type='KeypointMSELoss'),
                 loss_pose_l2norm: ConfigType = dict(type='pose_l2norm'),
                 loss_cosine_similarity: ConfigType = dict(type='cosine_similarity'),
                 loss_limb_length: ConfigType = dict(type='limb_length'),
                 loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss'),
                 loss_hmd: ConfigType = dict(type='MSELoss'),
                 loss_backbone_latant: ConfigType = dict(type='MSELoss'),
                 loss_backbone_heatmap: ConfigType = dict(type='KeypointMSELoss'),
                 # Stage 2 losses
                 loss_pose_l2norm_refined: ConfigType = dict(type='pose_l2norm'),
                 loss_bone_length: ConfigType = dict(type='bone_length_loss'),
                 loss_symmetry: ConfigType = dict(type='symmetry_loss'),
                 # Refinement params
                 refinement_hidden_size: int = 256,
                 refinement_num_stage: int = 1,
                 refinement_dropout: float = 0.5,
                 spatial_feat_dim: int = 64,
                 pose_feat_dim: int = 128,
                 kin_feat_dim: int = 64,
                 # Other
                 decoder: OptConfigType = None,
                 heatmap_decoder_type: str = 'original',
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)
        self.hm_iteration = 2000
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Stage 1 loss modules
        self.loss_module = MODELS.build(loss)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # Stage 2 loss modules
        self.loss_pose_l2norm_refined_module = MODELS.build(loss_pose_l2norm_refined)
        self.loss_bone_length_module = MODELS.build(loss_bone_length)
        self.loss_symmetry_module = MODELS.build(loss_symmetry)

        # ---- Stage 1 modules (same as Baseline) ----
        self.encoder = Encoder(num_classes=out_channels, output_size=64, hmd_info_size=9)

        self.heatmap_decoder_type = heatmap_decoder_type
        if heatmap_decoder_type == 'efficient':
            self.heatmap_decoder = EfficientHeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)
        else:
            self.heatmap_decoder = HeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)

        self.pose_decoder = LinearModel(
            input_size=64,
            num_classes=16,
            linear_size=512,
            num_stage=1,
            p_dropout=0.3
        )

        self.hmd_linear = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True)
        )

        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Deconv layers
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
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length.')
            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        self.add_deconv_layers = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # ---- Stage 2 modules (new) ----
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.in_channels, spatial_feat_dim),
            nn.ReLU(inplace=True),
        )

        self.pose_encoder_net = nn.Sequential(
            nn.Linear(out_channels * 3, pose_feat_dim),  # 16*3=48 → 128
            nn.ReLU(inplace=True),
        )

        num_bones = len(EGOPOSE_SKELETON)  # 15
        self.kin_encoder = nn.Sequential(
            nn.Linear(num_bones * 4, kin_feat_dim),  # 15*4=60 → 64
            nn.ReLU(inplace=True),
        )

        # Per-joint input: coarse_xyz(3) + spatial(64) + Z(64) + pose_ctx(128) + kin(64) = 323
        refinement_input_size = 3 + spatial_feat_dim + 64 + pose_feat_dim + kin_feat_dim
        self.refinement_mlp = RefinementMLP(
            input_size=refinement_input_size,
            hidden_size=refinement_hidden_size,
            num_stage=refinement_num_stage,
            p_dropout=refinement_dropout,
        )

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels, layer_out_channels, layer_kernel_sizes):
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(type='Conv2d', in_channels=in_channels,
                       out_channels=out_channels, kernel_size=kernel_size,
                       stride=1, padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels, layer_out_channels,
                            layer_kernel_sizes, layer_stride_sizes):
        layers = []
        for out_channels, kernel_size, stride_size in zip(
                layer_out_channels, layer_kernel_sizes, layer_stride_sizes):
            if kernel_size == 4:
                padding, output_padding = 1, 0
            elif kernel_size == 3:
                padding, output_padding = 1, 1
            elif kernel_size == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size}')
            cfg = dict(type='deconv', in_channels=in_channels,
                       out_channels=out_channels, kernel_size=kernel_size,
                       stride=stride_size, padding=padding,
                       output_padding=output_padding, bias=False)
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

    # -----------------------------------------------------------------
    # Forward: produces heatmap (same as Baseline)
    # -----------------------------------------------------------------
    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    # -----------------------------------------------------------------
    # Stage 2: Refinement
    # -----------------------------------------------------------------
    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent):
        """Stage 2: Refine coarse pose using spatial + kinematic features.

        Args:
            coarse_pose: [B, 16, 3] coarse 3D prediction from Stage 1
            heatmap: [B, 16, 47, 47] predicted heatmaps
            backbone_feat: [B, C, H, W] raw backbone feature map
            z_latent: [B, 64] latent vector from encoder

        Returns:
            refined_pose: [B, 16, 3]
        """
        B = coarse_pose.shape[0]
        K = coarse_pose.shape[1]  # 16

        # 1. Per-joint spatial features via grid sampling
        coords_2d, _ = soft_argmax_2d(heatmap.detach())  # [B, K, 2]
        # Convert [0,1] to [-1,1] for grid_sample
        grid = coords_2d * 2 - 1  # [B, K, 2]
        grid = grid.unsqueeze(1)   # [B, 1, K, 2]
        sampled = F.grid_sample(
            backbone_feat, grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        )  # [B, C, 1, K]
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, K, C]
        spatial_feat = self.spatial_proj(sampled)  # [B, K, spatial_feat_dim]

        # 2. Pose context encoding
        pose_flat = coarse_pose.reshape(B, -1)  # [B, 48]
        pose_feat = self.pose_encoder_net(pose_flat)  # [B, pose_feat_dim]

        # 3. Kinematic chain features
        kin_raw = compute_kinematic_features(coarse_pose)  # [B, 60]
        kin_feat = self.kin_encoder(kin_raw)  # [B, kin_feat_dim]

        # 4. Build per-joint input (broadcast global features)
        z_exp = z_latent.unsqueeze(1).expand(B, K, -1)         # [B, K, 64]
        pose_exp = pose_feat.unsqueeze(1).expand(B, K, -1)     # [B, K, pose_feat_dim]
        kin_exp = kin_feat.unsqueeze(1).expand(B, K, -1)       # [B, K, kin_feat_dim]

        joint_input = torch.cat([
            coarse_pose,    # [B, K, 3]
            spatial_feat,   # [B, K, spatial_feat_dim]
            z_exp,          # [B, K, 64]
            pose_exp,       # [B, K, pose_feat_dim]
            kin_exp,        # [B, K, kin_feat_dim]
        ], dim=-1)  # [B, K, 323]

        # 5. Shared refinement MLP
        joint_flat = joint_input.reshape(B * K, -1)  # [B*K, 323]
        delta = self.refinement_mlp(joint_flat)       # [B*K, 3]
        delta = delta.reshape(B, K, 3)                # [B, K, 3]

        return coarse_pose + delta

    # -----------------------------------------------------------------
    # Decode (for inference, same as Baseline + refinement)
    # -----------------------------------------------------------------
    def decode(self, batch_outputs, batch_data_samples,
               backbone_feat=None):
        """Decode keypoints from heatmap outputs.

        If backbone_feat is provided, also runs Stage 2 refinement.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}.')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)
            if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                batch_scores, batch_visibility = batch_scores
            else:
                batch_visibility = [None] * len(batch_keypoints)
        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints = []
            batch_scores = []
            batch_visibility = []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(
                    outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        # HMD info
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Stage 1: encode + predict
        z = self.encoder(batch_outputs.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = z + hmd_info_

        batch_3d_keypoints = self.pose_decoder(z_plus_hmd)
        generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
        hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)

        # Stage 2: refinement (if backbone features available)
        if backbone_feat is not None:
            coarse_pose = batch_3d_keypoints.reshape(-1, 16, 3)
            refined_pose = self.refine(
                coarse_pose, batch_outputs, backbone_feat, z)
            output_3d = refined_pose
        else:
            output_3d = batch_3d_keypoints

        # Pack results
        preds = []
        for (keypoints, kp3d, scores, visibility,
             gen_hm, hmd_rec) in zip(
                batch_keypoints, output_3d, batch_scores,
                batch_visibility, generated_heatmaps, hmd_recons):
            kp3d = kp3d.unsqueeze(dim=0)
            hmd_rec = hmd_rec.unsqueeze(dim=0)
            gen_hm = gen_hm.unsqueeze(dim=0)
            pred = InstanceData(
                keypoints=keypoints, keypoint_scores=scores,
                keypoint_3d=kp3d, generated_heatmap=gen_hm,
                hmd_recon=hmd_rec)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, output_3d

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------
    def predict(self, feats, batch_data_samples,
                test_cfg={}):
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

        preds, _ = self.decode(
            batch_heatmaps, batch_data_samples,
            backbone_feat=backbone_feat)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------
    def loss(self, feats, batch_data_samples, train_cfg={}):
        backbone_feat = feats[-1]  # [B, 2048, 8, 8]

        # Stage 1: heatmap forward
        pred_fields = self.forward(feats)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Stage 1: encode, predict
        z = self.encoder(pred_fields.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = z + hmd_info_

        coarse_pose = self.pose_decoder(z_plus_hmd)
        recon_heatmap = self.heatmap_decoder(z_plus_hmd)
        hmd_recon = preprocess_hmd_data_batch(coarse_pose)

        coarse_pose = coarse_pose.reshape(-1, 16, 3)

        # Stage 1 losses
        loss_pose_l2norm = self.loss_pose_l2norm_module(
            coarse_pose, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            coarse_pose, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(
            coarse_pose, gt_keypoint_3d)
        loss_heatmap_recon = self.loss_heatmap_recon_module(
            recon_heatmap, gt_heatmaps, keypoint_weights)
        loss_kpt = self.loss_module(
            pred_fields, gt_heatmaps, keypoint_weights)
        loss_hmd = self.loss_hmd_module(
            hmd_recon.to(torch.double), HMD_info.to(torch.double))

        # Stage 2: refinement
        refined_pose = self.refine(
            coarse_pose, pred_fields, backbone_feat, z)

        # Stage 2 losses
        loss_refined = self.loss_pose_l2norm_refined_module(
            refined_pose, gt_keypoint_3d)
        loss_bone = self.loss_bone_length_module(
            refined_pose, gt_keypoint_3d)
        loss_sym = self.loss_symmetry_module(refined_pose)

        # Aggregate losses
        losses = dict()
        losses.update(loss_pose_l2norm=torch.mean(loss_pose_l2norm))
        losses.update(loss_cosine_similarity=torch.mean(loss_cosine))
        losses.update(loss_limb_length=torch.mean(loss_limb))
        losses.update(loss_heatmap_recon=loss_heatmap_recon)
        losses.update(loss_hmd=loss_hmd)
        losses.update(loss_kpt=loss_kpt)
        # Stage 2
        losses.update(loss_pose_l2norm_refined=torch.mean(loss_refined))
        losses.update(loss_bone_length=torch.mean(loss_bone))
        losses.update(loss_symmetry=torch.mean(loss_sym))

        # Accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        self.hm_iteration += 1
        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta,
                                  *args, **kwargs):
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
