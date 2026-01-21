# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom EgoPose Lifting Head with Backbone Feature Fusion (Phase 5-C)

Architecture:
    Backbone feat [2048, 8, 8]
           │
           ├─────────────────────────────┐
           │                             │
           ↓ (Deconv)                    ↓ (GAP → FC)
    Heatmap [16, 47, 47]            Z_backbone [256]
           │                             │
           │  ← 2D 위치 (gradient 차단)   │  ← 3D depth (gradient 흐름)
           │                             │
           ↓ (soft_argmax)               │
    2D coords [32] + conf [16]           │
           │                             │
           └───────── Concat ────────────┘
                        ↓
               [32 + 16 + 256 + 9] = 313
                        ↓
                 Lifting Network
                        ↓
                    3D Pose

Key Design:
- coords_2d.detach(): Heatmap learns 2D only (from heatmap loss)
- Z_backbone: Backbone learns depth cues (from 3D loss)
- Complete role separation!

Comparison:
- Phase 5-B (Lifting only): 190mm - failed due to lack of depth info
- Phase 5-C (+ Backbone): Expected improvement with depth cues
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData, InstanceData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead

import numpy as np

OptIntSeq = Optional[Sequence[int]]


def soft_argmax_2d(heatmaps: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Differentiable 2D coordinate extraction from heatmaps.

    Args:
        heatmaps: [B, K, H, W] predicted heatmaps
        temperature: softmax temperature (lower = sharper)

    Returns:
        coords: [B, K, 2] normalized coordinates (0~1)
        confidence: [B, K] max heatmap values
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)  # [B, K, H*W]

    # Softmax over spatial dimensions (temperature scaling)
    heatmaps_soft = F.softmax(heatmaps_flat / temperature, dim=-1)  # [B, K, H*W]

    # Create coordinate grids (normalized 0~1)
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)

    # Create meshgrid and flatten
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx_flat = xx.reshape(-1)  # [H*W]
    yy_flat = yy.reshape(-1)  # [H*W]

    # Compute expected coordinates (weighted sum)
    x = (heatmaps_soft * xx_flat.view(1, 1, -1)).sum(dim=-1)  # [B, K]
    y = (heatmaps_soft * yy_flat.view(1, 1, -1)).sum(dim=-1)  # [B, K]

    coords = torch.stack([x, y], dim=-1)  # [B, K, 2]

    # Confidence = max value in heatmap
    confidence = heatmaps_flat.max(dim=-1)[0]  # [B, K]

    return coords, confidence


class BackboneEncoder(nn.Module):
    """
    Encode backbone features to latent vector for depth cues.

    Backbone feat [B, 2048, 8, 8] → Z_backbone [B, 256]
    """

    def __init__(self, in_channels: int = 2048, latent_dim: int = 256):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_feat: Tensor) -> Tensor:
        """
        Args:
            backbone_feat: [B, C, H, W]
        Returns:
            z_backbone: [B, latent_dim]
        """
        x = self.pool(backbone_feat)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        z = self.fc(x)  # [B, latent_dim]
        return z


class LiftingNetwork(nn.Module):
    """
    2D→3D Lifting Network with Backbone Features

    Input: 2D coords [B, 32] + confidence [B, 16] + Z_backbone [B, 256] + HMD [B, 9]
    Output: 3D pose [B, 16, 3]
    """

    def __init__(self,
                 num_joints: int = 16,
                 hmd_dim: int = 9,
                 backbone_latent_dim: int = 256,
                 hidden_dim: int = 1024,
                 num_blocks: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # Input: 2D coords (32) + confidence (16) + backbone (256) + HMD (9) = 313
        input_dim = num_joints * 2 + num_joints + backbone_latent_dim + hmd_dim
        output_dim = num_joints * 3

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _make_block(self, dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, coords_2d: Tensor, confidence: Tensor,
                z_backbone: Tensor, hmd_info: Tensor) -> Tensor:
        """
        Args:
            coords_2d: [B, 16, 2] normalized 2D coordinates
            confidence: [B, 16] heatmap confidence
            z_backbone: [B, 256] backbone latent (depth cues)
            hmd_info: [B, 9] HMD information

        Returns:
            pose_3d: [B, 16, 3]
        """
        B = coords_2d.size(0)

        # Flatten and concatenate inputs
        x = torch.cat([
            coords_2d.view(B, -1),      # [B, 32]
            confidence,                  # [B, 16]
            z_backbone,                  # [B, 256]
            hmd_info                     # [B, 9]
        ], dim=1)  # [B, 313]

        # Forward
        x = self.input_proj(x)  # [B, hidden_dim]

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        pose_3d = self.output_proj(x)  # [B, 48]
        pose_3d = pose_3d.view(B, -1, 3)  # [B, 16, 3]

        return pose_3d


@MODELS.register_module()
class CustomEgoposeLiftingBackboneFusionHead(BaseHead):
    """
    EgoPose Head with Soft-argmax Lifting + Backbone Feature Fusion (Phase 5-C)

    Key Design:
        1. Heatmap → soft_argmax → 2D coords (DETACHED - no gradient from 3D loss)
        2. Backbone → GAP → FC → Z_backbone (gradient flows from 3D loss)
        3. Lifting: 2D + conf + Z_backbone + HMD → 3D

    Role Separation:
        - Heatmap: 2D position only (learns from heatmap MSE + coord MSE)
        - Backbone: 3D depth cues (learns from 3D pose loss)
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
                 # Losses
                 loss: ConfigType = dict(type='KeypointMSELoss', loss_weight=1000),
                 loss_coord: ConfigType = dict(type='MSELoss', loss_weight=10.0),
                 loss_pose_l2norm: ConfigType = dict(type='pose_l2norm', loss_weight=1.0),
                 loss_cosine_similarity: ConfigType = dict(type='cosine_similarity', loss_weight=0.1),
                 loss_limb_length: ConfigType = dict(type='limb_length', loss_weight=0.25),
                 # Backbone encoder config
                 backbone_latent_dim: int = 256,
                 # Lifting network config
                 lifting_hidden_dim: int = 1024,
                 lifting_num_blocks: int = 2,
                 lifting_dropout: float = 0.3,
                 # Soft-argmax config
                 soft_argmax_temperature: float = 1.0,
                 # Decoder
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.soft_argmax_temperature = soft_argmax_temperature
        self.backbone_latent_dim = backbone_latent_dim

        # Loss modules
        self.loss_heatmap = MODELS.build(loss)
        self.loss_coord_module = MODELS.build(loss_coord)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)

        # Backbone encoder (for depth cues)
        self.backbone_encoder = BackboneEncoder(
            in_channels=in_channels,
            latent_dim=backbone_latent_dim
        )

        # Lifting network (2D + conf + backbone + HMD → 3D)
        self.lifting_network = LiftingNetwork(
            num_joints=out_channels,
            hmd_dim=9,
            backbone_latent_dim=backbone_latent_dim,
            hidden_dim=lifting_hidden_dim,
            num_blocks=lifting_num_blocks,
            dropout=lifting_dropout
        )

        # Decoder for evaluation
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Build deconv layers (backbone → heatmap)
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
            deconv_out = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()
            deconv_out = in_channels

        # Conv layers
        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length.')

            self.conv_layers = self._make_conv_layers(
                in_channels=deconv_out,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            conv_out = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()
            conv_out = deconv_out

        # Final layer (to heatmap channels)
        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=256,  # After add_deconv_layers
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
                layer_out_channels, layer_kernel_sizes, layer_stride_sizes):
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
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer='Linear', std=0.01)
        ]

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: backbone features → heatmap + backbone latent

        Args:
            feats: Tuple of backbone features

        Returns:
            heatmap: [B, K, 47, 47]
            backbone_feat: [B, C, H, W] for backbone encoder
        """
        backbone_feat = feats[-1]  # [B, 2048, 8, 8]

        # Heatmap generation
        x = self.deconv_layers(backbone_feat)
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        heatmap = self.final_layer(x)

        return heatmap, backbone_feat

    def forward_lifting(self, heatmaps: Tensor, backbone_feat: Tensor,
                        hmd_info: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Soft-argmax + Backbone Encoder + Lifting network

        Args:
            heatmaps: [B, K, H, W] predicted heatmaps
            backbone_feat: [B, C, H, W] backbone features
            hmd_info: [B, 9] HMD information

        Returns:
            pose_3d: [B, K, 3]
            coords_2d: [B, K, 2] normalized coordinates
            confidence: [B, K]
            z_backbone: [B, latent_dim]
        """
        # Soft-argmax: heatmap → 2D coordinates
        coords_2d, confidence = soft_argmax_2d(heatmaps, self.soft_argmax_temperature)

        # DETACH: 2D coords don't receive gradient from 3D loss
        # Heatmap learns 2D position only (from heatmap MSE + coord MSE)
        coords_2d_detached = coords_2d.detach()
        confidence_detached = confidence.detach()

        # Backbone encoder: extract depth cues
        # This DOES receive gradient from 3D loss
        z_backbone = self.backbone_encoder(backbone_feat)

        # Lifting: 2D + conf + backbone + HMD → 3D
        pose_3d = self.lifting_network(
            coords_2d_detached,
            confidence_detached,
            z_backbone,
            hmd_info
        )

        return pose_3d, coords_2d, confidence, z_backbone

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """
        Calculate losses.

        Losses:
            1. loss_heatmap: MSE(pred_heatmap, gt_heatmap) - dense supervision for 2D
            2. loss_coord: MSE(soft_argmax coords, gt_coords) - sub-pixel 2D accuracy
            3. loss_pose_l2norm: L2(pred_3d, gt_3d) - 3D accuracy (trains backbone encoder!)
            4. loss_cosine_similarity: cosine sim loss
            5. loss_limb_length: limb length consistency
        """
        # Forward: backbone → heatmap + backbone_feat
        pred_heatmaps, backbone_feat = self.forward(feats)

        # Get ground truth
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        gt_keypoint_3d = torch.cat([d.gt_instance_labels.keypoint3d for d in batch_data_samples])
        hmd_info = torch.cat([d.gt_instance_labels.hmd_info for d in batch_data_samples])

        # Get GT 2D coords (normalized 0~1) from gt_heatmaps using soft_argmax
        with torch.no_grad():
            gt_coords_2d, _ = soft_argmax_2d(gt_heatmaps, temperature=0.1)  # Sharp for GT

        # Lifting: heatmap → 2D → 3D (with backbone features!)
        pose_3d, coords_2d, confidence, z_backbone = self.forward_lifting(
            pred_heatmaps, backbone_feat, hmd_info
        )

        # === Losses ===
        losses = dict()

        # 1. Heatmap MSE loss (dense supervision for 2D)
        loss_heatmap = self.loss_heatmap(pred_heatmaps, gt_heatmaps, keypoint_weights)
        losses['loss_heatmap'] = loss_heatmap

        # 2. Coordinate loss (sub-pixel 2D accuracy)
        kpt_mask = keypoint_weights.view(-1, self.out_channels, 1) > 0
        loss_coord = F.mse_loss(
            coords_2d * kpt_mask,
            gt_coords_2d * kpt_mask,
            reduction='sum'
        ) / (kpt_mask.sum() + 1e-6)
        losses['loss_coord'] = loss_coord * self.loss_coord_module.loss_weight

        # 3. 3D pose L2 loss (this trains the backbone encoder!)
        loss_pose_l2norm = self.loss_pose_l2norm_module(pose_3d, gt_keypoint_3d)
        losses['loss_pose_l2norm'] = torch.mean(loss_pose_l2norm)

        # 4. Cosine similarity loss
        loss_cosine = self.loss_cosine_similarity_module(pose_3d, gt_keypoint_3d)
        losses['loss_cosine_similarity'] = torch.mean(loss_cosine)

        # 5. Limb length loss
        loss_limb = self.loss_limb_length_module(pose_3d, gt_keypoint_3d)
        losses['loss_limb_length'] = torch.mean(loss_limb)

        # Accuracy (for logging)
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            losses['acc_pose'] = torch.tensor(avg_acc, device=pred_heatmaps.device)

        return losses

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """
        Predict results from features.
        """
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps, _backbone_feat = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip)[0],
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
            backbone_feat = _backbone_feat
        else:
            batch_heatmaps, backbone_feat = self.forward(feats)

        # Get HMD info
        hmd_info = torch.cat([d.gt_instance_labels.hmd_info for d in batch_data_samples])

        # Lifting: heatmap → 2D → 3D
        pose_3d, coords_2d, confidence, z_backbone = self.forward_lifting(
            batch_heatmaps, backbone_feat, hmd_info
        )

        # Decode heatmaps to 2D keypoints (for evaluation)
        preds = self.decode(batch_heatmaps, batch_data_samples, pose_3d, coords_2d, confidence)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def decode(self,
               batch_heatmaps: Tensor,
               batch_data_samples: OptSampleList,
               pose_3d: Tensor,
               coords_2d: Tensor,
               confidence: Tensor) -> InstanceList:
        """
        Decode predictions to InstanceData format.
        """
        if self.decoder is None:
            raise RuntimeError('Decoder not set')

        # Use codec to decode heatmaps to pixel coordinates
        batch_keypoints, batch_scores = [], []
        batch_output_np = to_numpy(batch_heatmaps, unzip=True)

        for outputs in batch_output_np:
            keypoints, scores = self.decoder.decode(outputs)
            batch_keypoints.append(keypoints)
            if isinstance(scores, tuple):
                batch_scores.append(scores[0])
            else:
                batch_scores.append(scores)

        # Build predictions
        preds = []
        for i, (keypoints, scores) in enumerate(zip(batch_keypoints, batch_scores)):
            pred = InstanceData(
                keypoints=keypoints,
                keypoint_scores=scores,
                keypoint_3d=pose_3d[i:i+1].detach().cpu().numpy(),
                coords_2d_normalized=coords_2d[i:i+1].detach().cpu().numpy(),
                lifting_confidence=confidence[i:i+1].detach().cpu().numpy()
            )
            preds.append(pred)

        return preds

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
                    idx = int(k_parts[1])
                    if hasattr(self, 'conv_layers') and isinstance(self.conv_layers, nn.Sequential):
                        if idx < len(self.conv_layers):
                            k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                        else:
                            k_new = 'final_layer.' + k_parts[2]
                    else:
                        k_new = k
                else:
                    k_new = k
            else:
                k_new = k
            state_dict[prefix + k_new] = v
