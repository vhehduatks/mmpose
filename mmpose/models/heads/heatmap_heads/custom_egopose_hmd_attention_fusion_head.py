# Copyright (c) OpenMMLab. All rights reserved.
"""
HMD Attention Fusion Head

Uses cross-attention to fuse enhanced HMD info (12-dim) with visual features.
Instead of simple concatenation, HMD tokens attend to heatmap spatial features
to selectively weight relevant visual information based on HMD context.

Key idea:
    Baseline: HMD(12→36) → concat with hm_feat(256) → Linear → Z[64]
    This:     HMD(12) → 3 tokens (head, left, right heights) → CrossAttn(HMD, visual)
              → fused with GAP features → Z[64]

HMD token structure for both_from_ground (12-dim):
    - Base HMD (9-dim): right_local(3) + left_local(3) + distances(3)
    - Ground heights (3-dim): head_from_ground + left_hand_from_ground + right_hand_from_ground

    Tokenized as:
    - Token 1: Base HMD info (9-dim) → embedded to d_model
    - Token 2: Head height from ground (1-dim + context) → embedded to d_model
    - Token 3: Hand heights from ground (2-dim + context) → embedded to d_model
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


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


def preprocess_hmd_data_batch(p3d):
    """Compute HMD reconstruction from predicted 3D pose (9-dim only)."""
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


class HMDAttentionEncoder(nn.Module):
    """Encoder with cross-attention-based HMD fusion.

    Architecture:
        Heatmap [B, 16, 47, 47]
          → Conv1→Conv2→Conv3 → [B, 256, 6, 6]
          → Reshape → [B, 36, 256] (spatial tokens)
          → + Positional Embedding

        HMD [B, hmd_info_size]
          → Tokenize: [B, n_hmd_tokens, d_model]
          → CrossAttn(HMD_tokens as Query, Visual_tokens as KV)
          → [B, n_hmd_tokens, d_model]

        Fusion:
          → GAP visual [B, 256]
          → HMD attention output [B, n_hmd_tokens * d_model]
          → Concat + Project → Z[64]
    """

    def __init__(self, num_classes=16, output_size=64, hmd_info_size=12,
                 d_model=64, n_attn_heads=4, attn_dropout=0.1):
        super(HMDAttentionEncoder, self).__init__()
        self.conv_dim = 256
        self.d_model = d_model
        self.hmd_info_size = hmd_info_size

        # Conv layers (same as baseline)
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, self.conv_dim, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)

        # GAP (baseline path)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Visual token projection to d_model
        self.visual_proj = nn.Linear(self.conv_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 36, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # HMD tokenization
        # For both_from_ground (12-dim): base(9) + head_height(1) + hand_heights(2)
        # Tokenize into 3 semantic tokens
        self.n_hmd_tokens = 3

        # Token embeddings for different HMD components
        # Token 1: Base HMD info (9-dim)
        self.base_hmd_embed = nn.Sequential(
            nn.Linear(9, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # Token 2: Head height from ground (1-dim for 10/11/12-dim modes)
        # Token 3: Hand heights or torso info (remaining dims)
        extra_dims = hmd_info_size - 9
        if extra_dims > 0:
            # Head height token
            self.head_height_embed = nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True),
            )
            # Hand/extra token (remaining dims)
            remaining_dims = max(1, extra_dims - 1)
            self.extra_embed = nn.Sequential(
                nn.Linear(remaining_dims, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.head_height_embed = None
            self.extra_embed = None

        # Cross-attention: HMD tokens attend to visual tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm_q = nn.LayerNorm(d_model)
        self.attn_norm_kv = nn.LayerNorm(d_model)

        # Output projection
        # GAP(256) + HMD_attn_output(n_tokens * d_model) → output_size
        hmd_output_dim = self.n_hmd_tokens * d_model if extra_dims > 0 else d_model
        self.output_proj = nn.Sequential(
            nn.Linear(self.conv_dim + hmd_output_dim, output_size),
            nn.LeakyReLU(0.2),
        )

        # Fallback for no HMD case
        self.output_proj_no_hmd = nn.Sequential(
            nn.Linear(self.conv_dim, output_size),
            nn.LeakyReLU(0.2),
        )

    def tokenize_hmd(self, hmd):
        """Convert HMD info to tokens.

        Args:
            hmd: [B, hmd_info_size] HMD info

        Returns:
            tokens: [B, n_tokens, d_model]
        """
        B = hmd.shape[0]
        tokens = []

        # Token 1: Base HMD (first 9 dims)
        base_hmd = hmd[:, :9]
        token1 = self.base_hmd_embed(base_hmd)  # [B, d_model]
        tokens.append(token1)

        # Additional tokens for enhanced HMD
        if self.hmd_info_size > 9 and self.head_height_embed is not None:
            # Token 2: Head height (dim 9)
            head_height = hmd[:, 9:10]  # [B, 1]
            token2 = self.head_height_embed(head_height)  # [B, d_model]
            tokens.append(token2)

            # Token 3: Remaining dims (10:end)
            if self.hmd_info_size > 10:
                extra = hmd[:, 10:]  # [B, remaining]
            else:
                # Pad with head height if only 10-dim
                extra = head_height
            token3 = self.extra_embed(extra)  # [B, d_model]
            tokens.append(token3)

        # Stack tokens: [B, n_tokens, d_model]
        tokens = torch.stack(tokens, dim=1)
        return tokens

    def forward(self, hm, hmd=None):
        # Conv layers
        hm = self.conv1(hm)
        hm = self.lrelu1(hm)
        hm = self.conv2(hm)
        hm = self.lrelu2(hm)
        hm = self.conv3(hm)
        hm = self.lrelu3(hm)
        # hm: [B, 256, 6, 6]

        B = hm.shape[0]

        # GAP features
        hm_gap = self.avr_pool(hm).reshape(B, self.conv_dim)

        if hmd is not None:
            # Visual tokens
            spatial_tokens = hm.reshape(B, self.conv_dim, -1).permute(0, 2, 1)  # [B, 36, 256]
            spatial_tokens = self.visual_proj(spatial_tokens)  # [B, 36, d_model]
            spatial_tokens = spatial_tokens + self.pos_embed

            # HMD tokens
            hmd_tokens = self.tokenize_hmd(hmd)  # [B, n_tokens, d_model]

            # Cross-attention: HMD queries visual features
            q = self.attn_norm_q(hmd_tokens)
            kv = self.attn_norm_kv(spatial_tokens)
            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            # attn_out: [B, n_tokens, d_model]

            # Flatten attention output
            hmd_fused = attn_out.reshape(B, -1)  # [B, n_tokens * d_model]

            # Combine GAP + HMD attention
            combined = torch.cat([hm_gap, hmd_fused], dim=1)
            z = self.output_proj(combined)
        else:
            z = self.output_proj_no_hmd(hm_gap)

        return z


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


@MODELS.register_module()
class CustomEgoposeHMDAttentionFusionHead(BaseHead):
    """Baseline head with attention-based HMD fusion.

    Uses cross-attention to fuse enhanced HMD info with visual features,
    allowing the model to selectively weight visual information based on
    HMD context (e.g., ground heights).

    Args:
        hmd_info_size (int): Size of HMD info. Default: 12 (both_from_ground).
        d_model (int): Attention dimension. Default: 64.
        n_attn_heads (int): Number of attention heads. Default: 4.
        attn_dropout (float): Attention dropout. Default: 0.1.
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 hmd_info_size: int = 12,
                 d_model: int = 64,
                 n_attn_heads: int = 4,
                 attn_dropout: float = 0.1,
                 deconv_out_channels: OptIntSeq = (256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4),
                 deconv_stride_sizes: OptIntSeq = (2, 2),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(type='KeypointMSELoss'),
                 loss_pose_l2norm: ConfigType = dict(type='pose_l2norm'),
                 loss_cosine_similarity: ConfigType = dict(type='cosine_similarity'),
                 loss_limb_length: ConfigType = dict(type='limb_length'),
                 loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss'),
                 loss_hmd: ConfigType = dict(type='MSELoss'),
                 decoder: OptConfigType = None,
                 heatmap_decoder_type: str = 'original',
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)
        self.hm_iteration = 2000
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hmd_info_size = hmd_info_size

        # Loss modules
        self.loss_module = MODELS.build(loss)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)
        self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon)
        self.loss_hmd_module = MODELS.build(loss_hmd)

        # Encoder with attention-based HMD fusion
        self.encoder = HMDAttentionEncoder(
            num_classes=out_channels,
            output_size=64,
            hmd_info_size=hmd_info_size,
            d_model=d_model,
            n_attn_heads=n_attn_heads,
            attn_dropout=attn_dropout,
        )

        # Heatmap decoder
        self.heatmap_decoder_type = heatmap_decoder_type
        if heatmap_decoder_type == 'efficient':
            self.heatmap_decoder = EfficientHeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)
        else:
            self.heatmap_decoder = HeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)

        # Pose decoder
        self.pose_decoder = LinearModel(
            input_size=64,
            num_classes=16,
            linear_size=512,
            num_stage=1,
            p_dropout=0.3
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

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def decode(self, batch_outputs, batch_data_samples):
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

        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Encode with attention-based HMD fusion
        z = self.encoder(batch_outputs.to(torch.float32), HMD_info.to(torch.float32))

        batch_3d_keypoints = self.pose_decoder(z)
        generated_heatmaps = self.heatmap_decoder(z)
        hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)

        preds = []
        for (keypoints, kp3d, scores, visibility,
             gen_hm, hmd_rec) in zip(
                batch_keypoints, batch_3d_keypoints, batch_scores,
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

        return preds, batch_3d_keypoints

    def predict(self, feats, batch_data_samples, test_cfg={}):
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

    def loss(self, feats, batch_data_samples, train_cfg={}):
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

        # Encode with attention-based HMD fusion
        z = self.encoder(pred_fields.to(torch.float32), HMD_info.to(torch.float32))

        pred_pose_3d = self.pose_decoder(z)
        recon_heatmap = self.heatmap_decoder(z)
        hmd_recon = preprocess_hmd_data_batch(pred_pose_3d)

        pred_pose_3d = pred_pose_3d.reshape(-1, 16, 3)

        # Losses
        loss_pose_l2norm = self.loss_pose_l2norm_module(
            pred_pose_3d, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            pred_pose_3d, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(
            pred_pose_3d, gt_keypoint_3d)
        loss_heatmap_recon = self.loss_heatmap_recon_module(
            recon_heatmap, gt_heatmaps, keypoint_weights)
        loss_kpt = self.loss_module(
            pred_fields, gt_heatmaps, keypoint_weights)

        # HMD loss only uses first 9 dims
        loss_hmd = self.loss_hmd_module(
            hmd_recon.to(torch.double), HMD_info[:, :9].to(torch.double))

        losses = dict()
        losses.update(loss_pose_l2norm=torch.mean(loss_pose_l2norm))
        losses.update(loss_cosine_similarity=torch.mean(loss_cosine))
        losses.update(loss_limb_length=torch.mean(loss_limb))
        losses.update(loss_heatmap_recon=loss_heatmap_recon)
        losses.update(loss_hmd=loss_hmd)
        losses.update(loss_kpt=loss_kpt)

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
