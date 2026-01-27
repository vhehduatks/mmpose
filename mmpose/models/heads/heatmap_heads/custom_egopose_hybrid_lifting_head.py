# Copyright (c) OpenMMLab. All rights reserved.
"""
Hybrid Lifting Head: Baseline Conv Encoder + Attention Refinement

핵심 아이디어:
- Baseline의 검증된 Conv Encoder 재사용 (안정성)
- Z를 관절별로 분해하여 Self-Attention 적용 (관절 관계 모델링)
- HMD 정보는 Cross-Attention으로 융합 (선택적 참조)

구조:
    Heatmap [16, 47, 47]
         ↓
    Conv Encoder (Baseline 동일) → Z [64]
         ↓
    Z reshape → [16, 4] (관절당 4-dim)
         ↓
    Joint Embedding → [16, D]
         ↓
    Self-Attention (관절 간 관계)
         ↓
    HMD Cross-Attention (선택적 HMD 참조)
         ↓
    Output Head → [16, 3] (3D pose)
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData, InstanceData
from torch import Tensor

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead

import numpy as np
from .blocks import HeatmapDecoder

OptIntSeq = Optional[Sequence[int]]


class Encoder(nn.Module):
    """Baseline Conv Encoder: Heatmap → Latent Z"""
    def __init__(self, num_classes=16, output_size=64, hmd_info_size=9):
        super().__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear2_ = nn.Linear(256, output_size)
        self.lrelu5 = nn.LeakyReLU(0.2)

    def forward(self, hm):
        hm = self.conv1(hm)
        hm = self.lrelu1(hm)
        hm = self.conv2(hm)
        hm = self.lrelu2(hm)
        hm = self.conv3(hm)
        hm = self.lrelu3(hm)

        hm_avgpool = self.avr_pool(hm).view(-1, 256)
        x = self.linear2_(hm_avgpool)
        x = self.lrelu5(x)

        return x


class HybridLiftingModule(nn.Module):
    """Baseline Z + Attention Refinement for 3D Pose

    Z [64] → reshape [16, 4] → Self-Attn → Cross-Attn(HMD) → 3D Pose
    """

    def __init__(self, num_joints=16, latent_dim=64, joint_dim=64,
                 num_heads=4, num_self_attn_layers=2, dropout=0.1):
        super().__init__()

        self.num_joints = num_joints
        # Z를 관절별로 분해: 64 = 16 * 4
        self.latent_per_joint = latent_dim // num_joints  # 4

        # Joint embedding: 4 → joint_dim
        self.joint_embed = nn.Linear(self.latent_per_joint, joint_dim)

        # Positional encoding for 16 joints
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, joint_dim) * 0.02)

        # Self-Attention layers (관절 간 관계)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=joint_dim, nhead=num_heads,
                dim_feedforward=joint_dim * 4, dropout=dropout,
                batch_first=True, norm_first=True
            ) for _ in range(num_self_attn_layers)
        ])

        # HMD embedding: 9 → 3 tokens (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)
        )

        # Cross-Attention: joints query HMD
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(joint_dim)

        # Output projection: joint_dim → 3
        self.output_proj = nn.Sequential(
            nn.Linear(joint_dim, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, 3)
        )

    def forward(self, z, hmd_info):
        B = z.size(0)

        # Z → joint tokens: [B, 64] → [B, 16, 4] → [B, 16, D]
        z_joints = z.view(B, self.num_joints, self.latent_per_joint)  # [B, 16, 4]
        joint_tokens = self.joint_embed(z_joints)  # [B, 16, D]
        joint_tokens = joint_tokens + self.pos_embed

        # Self-Attention (관절 간 관계)
        for layer in self.self_attn_layers:
            joint_tokens = layer(joint_tokens)

        # HMD tokens: [B, 9] → [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross-Attention (joints ← HMD)
        cross_out, _ = self.cross_attn(
            query=joint_tokens, key=hmd_tokens, value=hmd_tokens
        )
        joint_tokens = self.cross_norm(joint_tokens + cross_out)

        # Output: [B, 16, 3]
        pose_3d = self.output_proj(joint_tokens)

        return pose_3d


def preprocess_hmd_data_batch(p3d):
    """Calculate HMD info from 3D pose for reconstruction loss"""
    if not isinstance(p3d, torch.Tensor):
        p3d = torch.tensor(p3d, dtype=torch.float32)

    head = p3d[:, 0]
    right_hand = p3d[:, 7]
    left_hand = p3d[:, 4]

    midpoint = (right_hand + left_hand) / 2
    z_axis = midpoint - head
    z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-6)

    hand_vector = right_hand - left_hand
    x_axis = torch.cross(z_axis, hand_vector, dim=1)
    x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-6)

    y_axis = torch.cross(z_axis, x_axis, dim=1)

    rotation_matrices = torch.stack((x_axis, y_axis, z_axis), dim=2)

    right_local = torch.bmm(rotation_matrices.transpose(1, 2), (right_hand - head).unsqueeze(2)).squeeze(2)
    left_local = torch.bmm(rotation_matrices.transpose(1, 2), (left_hand - head).unsqueeze(2)).squeeze(2)

    hand_distance = torch.norm(right_local - left_local, dim=1)
    right_distance = torch.norm(right_local, dim=1)
    left_distance = torch.norm(left_local, dim=1)

    preprocessed_hmd = torch.cat([
        right_local, left_local,
        hand_distance.unsqueeze(1), right_distance.unsqueeze(1), left_distance.unsqueeze(1)
    ], dim=1)

    return preprocessed_hmd


@MODELS.register_module()
class CustomEgoposeHybridLiftingHead(BaseHead):
    """Hybrid Lifting Head: Baseline Conv Encoder + Attention Refinement

    Args:
        in_channels: Backbone output channels (2048 for ResNet-101)
        out_channels: Number of joints (16)
        joint_dim: Dimension of joint tokens (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_self_attn_layers: Number of self-attention layers (default: 2)
        dropout: Dropout rate (default: 0.1)
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
                 # Hybrid Lifting params
                 joint_dim: int = 64,
                 num_heads: int = 4,
                 num_self_attn_layers: int = 2,
                 dropout: float = 0.1,
                 # Losses
                 loss: ConfigType = dict(type='KeypointMSELoss'),
                 loss_pose_l2norm: ConfigType = dict(type='pose_l2norm'),
                 loss_cosine_similarity: ConfigType = dict(type='cosine_similarity'),
                 loss_limb_length: ConfigType = dict(type='limb_length'),
                 loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss'),
                 loss_hmd: ConfigType = dict(type='MSELoss'),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build loss modules
        self.loss_module = MODELS.build(loss)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # ===== Baseline Components =====
        # Conv Encoder: Heatmap → Z [64]
        self.encoder = Encoder(num_classes=out_channels, output_size=64, hmd_info_size=9)

        # Heatmap Decoder: Z → Heatmap reconstruction
        self.heatmap_decoder = HeatmapDecoder(
            num_classes=out_channels, heatmap_resolution=47, input_size=64)

        # ===== Hybrid Lifting Module =====
        self.hybrid_lifting = HybridLiftingModule(
            num_joints=out_channels,
            latent_dim=64,
            joint_dim=joint_dim,
            num_heads=num_heads,
            num_self_attn_layers=num_self_attn_layers,
            dropout=dropout
        )

        # Codec
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # ===== Deconv layers for 2D heatmap =====
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

        # Upsample to 47x47
        self.add_deconv_layers = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
                layer_out_channels, layer_kernel_sizes, layer_stride_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
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
        init_cfg = [
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward: backbone feat → 2D heatmap"""
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def decode(self, batch_outputs: Union[Tensor, Tuple[Tensor]],
               batch_data_samples: OptSampleList) -> InstanceList:
        """Decode keypoints from heatmap outputs."""

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError('Decoder has not been set')

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
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        # Get HMD info
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Encode heatmap → Z
        z = self.encoder(batch_outputs.to(torch.float32))

        # Hybrid Lifting: Z + HMD → 3D pose
        batch_3d_keypoints = self.hybrid_lifting(z, HMD_info.to(torch.float32))

        # Heatmap reconstruction
        generated_heatmaps = self.heatmap_decoder(z)

        # HMD reconstruction from predicted 3D pose
        hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)

        preds = []
        for keypoints, keypoint_3d, scores, visibility, generated_heatmap, hmd_recon in zip(
                batch_keypoints, batch_3d_keypoints, batch_scores,
                batch_visibility, generated_heatmaps, hmd_recons):
            keypoint_3d = keypoint_3d.unsqueeze(dim=0)
            hmd_recon = hmd_recon.unsqueeze(dim=0)
            generated_heatmap = generated_heatmap.unsqueeze(dim=0)
            pred = InstanceData(
                keypoints=keypoints,
                keypoint_scores=scores,
                keypoint_3d=keypoint_3d,
                generated_heatmap=generated_heatmap,
                hmd_recon=hmd_recon
            )
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, batch_3d_keypoints

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features."""

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
        else:
            batch_heatmaps = self.forward(feats)

        preds, _ = self.decode(batch_heatmaps, batch_data_samples)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses."""

        # Forward: backbone → 2D heatmap
        pred_fields = self.forward(feats)

        # Ground truth
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Decode and get predictions
        pred, pred_batch_3d_keypoints = self.decode(pred_fields, batch_data_samples)

        # Reconstructed outputs
        pred_recon_heatmap = torch.cat([p.generated_heatmap for p in pred])
        pred_recon_hmd = torch.cat([p.hmd_recon for p in pred])

        # Reshape 3D keypoints
        pred_batch_3d_keypoints = pred_batch_3d_keypoints.view(-1, 16, 3)

        # ===== Calculate losses =====
        losses = dict()

        # 2D heatmap loss
        loss_2dkpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
        losses['loss_kpt'] = loss_2dkpt

        # 3D pose losses
        loss_pose_l2norm = self.loss_pose_l2norm_module(pred_batch_3d_keypoints, gt_keypoint_3d)
        loss_cosine_similarity = self.loss_cosine_similarity_module(pred_batch_3d_keypoints, gt_keypoint_3d)
        loss_limb_length = self.loss_limb_length_module(pred_batch_3d_keypoints, gt_keypoint_3d)

        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine_similarity)
        losses['loss_limb_length'] = torch.mean(loss_limb_length)

        # Heatmap reconstruction loss
        loss_heatmap_recon = self.loss_heatmap_recon_module(pred_recon_heatmap, gt_heatmaps, keypoint_weights)
        losses['loss_heatmap_recon'] = loss_heatmap_recon

        # HMD reconstruction loss
        loss_hmd = self.loss_hmd_module(pred_recon_hmd.to(torch.double), HMD_info.to(torch.double))
        losses['loss_hmd'] = loss_hmd

        # Accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses['acc_pose'] = acc_pose

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
