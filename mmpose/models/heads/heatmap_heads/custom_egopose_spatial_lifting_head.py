# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom EgoPose Spatial Lifting Head (Phase 5-D)

Key Innovation: Grid Sampling for per-joint depth feature extraction

Architecture:
    Backbone feat [B, 2048, 8, 8]
           │
           ├─────────────────────────────────────┐
           │                                     │
           ↓ (Deconv)                            │
    Heatmap [B, 16, 47, 47]                      │
           │                                     │
           ↓ (soft_argmax)                       │
    coords_2d [B, 16, 2]                         │
           │                                     │
           │ (detach)                            │
           ↓                                     ↓
    coords_2d_detached ──────────→ grid_sample(backbone, coords)
                                             │
                                             ↓
                                  joint_features [B, 16, 2048]
                                             │
                                             ↓ (FC)
                                  joint_depth [B, 16, 64]
                                             │
           ┌─────────────────────────────────┘
           │
           ↓
    Concat [coords(2) + conf(1) + depth(64) + hmd_shared] per joint
           │
           ↓
    Per-Joint MLP → Joint 3D [B, 16, 3]

Key Design:
- coords_2d.detach(): Heatmap learns 2D only
- grid_sample: Extract backbone features AT joint locations
- Per-joint depth: Each joint learns its own depth from local features
- Spatial information preserved!

vs Phase 5-C (GAP):
- 5-C: Global pooling loses spatial info → failed (176mm)
- 5-D: Per-joint sampling preserves spatial info → expected improvement
"""

from typing import List, Optional, Sequence, Tuple, Union

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


class SpatialDepthExtractor(nn.Module):
    """
    Extract per-joint features from backbone using grid_sample.

    Key: Uses 2D coordinates to sample backbone features at joint locations.
    This preserves spatial information that GAP loses.

    Backbone feat [B, C, H, W] + coords [B, K, 2] → joint_features [B, K, out_dim]
    """

    def __init__(self,
                 in_channels: int = 2048,
                 out_channels: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        # Per-joint feature projection
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_feat: Tensor, coords_2d: Tensor) -> Tensor:
        """
        Args:
            backbone_feat: [B, C, H, W] backbone feature map (e.g., [B, 2048, 8, 8])
            coords_2d: [B, K, 2] normalized coordinates in range [0, 1]

        Returns:
            joint_features: [B, K, out_channels] per-joint depth features
        """
        B, C, H, W = backbone_feat.shape
        B, K, _ = coords_2d.shape

        # Step 1: Convert coords from [0,1] to [-1,1] for grid_sample
        # grid_sample convention: (-1,-1) = top-left, (+1,+1) = bottom-right
        coords_normalized = coords_2d * 2 - 1  # [B, K, 2]

        # Step 2: Reshape for grid_sample
        # grid_sample expects grid of shape [B, H_out, W_out, 2]
        # We sample K points, so: [B, K, 1, 2]
        coords_grid = coords_normalized.unsqueeze(2)  # [B, K, 1, 2]

        # Step 3: Sample backbone features at joint locations
        # Uses bilinear interpolation for sub-pixel accuracy
        sampled = F.grid_sample(
            backbone_feat,      # [B, C, H, W]
            coords_grid,        # [B, K, 1, 2]
            mode='bilinear',
            padding_mode='border',  # Use border values for out-of-bounds
            align_corners=True
        )  # Output: [B, C, K, 1]

        # Step 4: Reshape to [B, K, C]
        sampled = sampled.squeeze(-1)  # [B, C, K]
        sampled = sampled.permute(0, 2, 1)  # [B, K, C]

        # Step 5: Project to output dimension
        joint_features = self.fc(sampled)  # [B, K, out_channels]

        return joint_features


class PerJointLiftingNetwork(nn.Module):
    """
    Per-joint 2D→3D lifting with spatial depth features.

    Each joint gets: coords(2) + confidence(1) + depth_feature(64) = 67
    Plus shared HMD info.

    Architecture:
        Per-joint features → shared MLP → 3D coordinates
    """

    def __init__(self,
                 num_joints: int = 16,
                 depth_dim: int = 64,
                 hmd_dim: int = 9,
                 hidden_dim: int = 256,
                 num_blocks: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.num_joints = num_joints
        self.depth_dim = depth_dim

        # Per-joint input: coords(2) + conf(1) + depth(64) = 67
        per_joint_dim = 2 + 1 + depth_dim

        # HMD embedding (shared across joints)
        self.hmd_embed = nn.Sequential(
            nn.Linear(hmd_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        # Per-joint input + HMD embedding
        input_dim = per_joint_dim + 64  # 67 + 64 = 131

        # Shared MLP for all joints (weight sharing)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ))

        # Output: 3D coordinates per joint
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self,
                coords_2d: Tensor,
                confidence: Tensor,
                joint_depth: Tensor,
                hmd_info: Tensor) -> Tensor:
        """
        Args:
            coords_2d: [B, K, 2] 2D coordinates (detached)
            confidence: [B, K] heatmap confidence (detached)
            joint_depth: [B, K, depth_dim] per-joint depth features
            hmd_info: [B, 9] HMD information

        Returns:
            pose_3d: [B, K, 3] 3D pose
        """
        B, K, _ = coords_2d.shape

        # Embed HMD info (shared across all joints)
        hmd_embed = self.hmd_embed(hmd_info)  # [B, 64]
        hmd_embed = hmd_embed.unsqueeze(1).expand(-1, K, -1)  # [B, K, 64]

        # Concatenate per-joint features
        # coords: [B, K, 2], conf: [B, K, 1], depth: [B, K, 64], hmd: [B, K, 64]
        x = torch.cat([
            coords_2d,                    # [B, K, 2]
            confidence.unsqueeze(-1),     # [B, K, 1]
            joint_depth,                  # [B, K, 64]
            hmd_embed                     # [B, K, 64]
        ], dim=-1)  # [B, K, 131]

        # Reshape to process all joints together: [B*K, 131]
        x = x.view(B * K, -1)

        # Forward through shared MLP
        x = self.input_proj(x)  # [B*K, hidden_dim]

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        # Output 3D coordinates
        pose_3d = self.output_proj(x)  # [B*K, 3]
        pose_3d = pose_3d.view(B, K, 3)  # [B, K, 3]

        return pose_3d


@MODELS.register_module()
class CustomEgoposeSpatialLiftingHead(BaseHead):
    """
    EgoPose Head with Spatial Depth Extraction via Grid Sampling (Phase 5-D)

    Key Innovation:
        - Uses grid_sample to extract backbone features AT joint locations
        - Each joint gets its own depth feature (not global pooling)
        - Preserves spatial information for depth learning

    Gradient Flow:
        - coords_2d: DETACHED - heatmap learns 2D only
        - joint_depth: gradient flows to backbone via grid_sample
        - 3D loss trains backbone to produce good depth features at joint locations
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
                 # Spatial depth extractor config
                 depth_feature_dim: int = 64,
                 # Lifting network config
                 lifting_hidden_dim: int = 256,
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
        self.depth_feature_dim = depth_feature_dim

        # Loss modules
        self.loss_heatmap = MODELS.build(loss)
        self.loss_coord_module = MODELS.build(loss_coord)
        self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
        self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
        self.loss_limb_length_module = MODELS.build(loss_limb_length)

        # Spatial depth extractor (NEW: replaces BackboneEncoder)
        self.spatial_depth_extractor = SpatialDepthExtractor(
            in_channels=in_channels,
            out_channels=depth_feature_dim,
            dropout=lifting_dropout
        )

        # Per-joint lifting network
        self.lifting_network = PerJointLiftingNetwork(
            num_joints=out_channels,
            depth_dim=depth_feature_dim,
            hmd_dim=9,
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
                in_channels=256,
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

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int],
                            layer_stride_sizes: Sequence[int]) -> nn.Module:
        """Create deconv layers."""
        layers = []
        for out_channels, kernel_size, stride in zip(
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

            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create conv layers."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
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
        Forward pass: backbone features → heatmap + backbone_feat

        Args:
            feats: Tuple of backbone features

        Returns:
            heatmap: [B, K, 47, 47]
            backbone_feat: [B, C, H, W] for spatial depth extraction
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
        Soft-argmax + Spatial Depth Extraction + Lifting

        Args:
            heatmaps: [B, K, H, W] predicted heatmaps
            backbone_feat: [B, C, H, W] backbone features
            hmd_info: [B, 9] HMD information

        Returns:
            pose_3d: [B, K, 3]
            coords_2d: [B, K, 2] normalized coordinates
            confidence: [B, K]
            joint_depth: [B, K, depth_dim]
        """
        # Soft-argmax: heatmap → 2D coordinates
        coords_2d, confidence = soft_argmax_2d(heatmaps, self.soft_argmax_temperature)

        # DETACH: 2D coords don't receive gradient from 3D loss
        coords_2d_detached = coords_2d.detach()
        confidence_detached = confidence.detach()

        # Spatial depth extraction: sample backbone at joint locations
        # This DOES receive gradient from 3D loss!
        joint_depth = self.spatial_depth_extractor(backbone_feat, coords_2d_detached)

        # Lifting: 2D + conf + depth + HMD → 3D
        pose_3d = self.lifting_network(
            coords_2d_detached,
            confidence_detached,
            joint_depth,
            hmd_info
        )

        return pose_3d, coords_2d, confidence, joint_depth

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """
        Calculate losses.

        Losses:
            1. loss_heatmap: MSE(pred_heatmap, gt_heatmap) - dense 2D supervision
            2. loss_coord: MSE(soft_argmax coords, gt_coords) - sub-pixel 2D
            3. loss_pose_l2norm: L2(pred_3d, gt_3d) - 3D (trains spatial extractor!)
            4. loss_cosine_similarity
            5. loss_limb_length
        """
        # Forward: backbone → heatmap + backbone_feat
        pred_heatmaps, backbone_feat = self.forward(feats)

        # Get ground truth
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        gt_keypoint_3d = torch.cat([d.gt_instance_labels.keypoint3d for d in batch_data_samples])
        hmd_info = torch.cat([d.gt_instance_labels.hmd_info for d in batch_data_samples])

        # Get GT 2D coords from gt_heatmaps
        with torch.no_grad():
            gt_coords_2d, _ = soft_argmax_2d(gt_heatmaps, temperature=0.1)

        # Lifting with spatial depth features
        pose_3d, coords_2d, confidence, joint_depth = self.forward_lifting(
            pred_heatmaps, backbone_feat, hmd_info
        )

        # === Losses ===
        losses = dict()

        # 1. Heatmap MSE loss
        loss_heatmap = self.loss_heatmap(pred_heatmaps, gt_heatmaps, keypoint_weights)
        losses['loss_heatmap'] = loss_heatmap

        # 2. Coordinate loss
        kpt_mask = keypoint_weights.view(-1, self.out_channels, 1) > 0
        loss_coord = F.mse_loss(
            coords_2d * kpt_mask,
            gt_coords_2d * kpt_mask,
            reduction='sum'
        ) / (kpt_mask.sum() + 1e-6)
        losses['loss_coord'] = loss_coord * self.loss_coord_module.loss_weight

        # 3. 3D pose L2 loss (trains spatial depth extractor!)
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
        """Predict results from features."""
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_heatmaps, _backbone_feat = self.forward(_feats)
            _batch_heatmaps_flip, _ = self.forward(_feats_flip)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) / 2.0
            backbone_feat = _backbone_feat
        else:
            batch_heatmaps, backbone_feat = self.forward(feats)

        # Get HMD info
        hmd_info = torch.cat([d.gt_instance_labels.hmd_info for d in batch_data_samples])

        # Lifting
        pose_3d, coords_2d, confidence, joint_depth = self.forward_lifting(
            batch_heatmaps, backbone_feat, hmd_info
        )

        preds = self.decode((batch_heatmaps, pose_3d, coords_2d))

        if test_cfg.get('output_heatmaps', False):
            for pred, heatmap in zip(preds, batch_heatmaps.detach().cpu().numpy()):
                pred.heatmaps = heatmap
                pred.heatmap_size = heatmap.shape[1:3]

        return preds

    def decode(self, batch_outputs: Tuple[Tensor, Tensor, Tensor]) -> List[InstanceData]:
        """Decode heatmaps and 3D poses into keypoint predictions."""
        batch_heatmaps, batch_pose_3d, batch_coords_2d = batch_outputs

        preds = []
        batch_size = batch_heatmaps.shape[0]

        for i in range(batch_size):
            heatmaps = batch_heatmaps[i].cpu().numpy()
            pose_3d = batch_pose_3d[i].cpu().numpy()
            coords_2d = batch_coords_2d[i].cpu().numpy()

            # Get 2D keypoints from heatmaps
            if self.decoder is not None:
                keypoints, scores = self.decoder.decode(heatmaps)
            else:
                # Fallback: use soft_argmax coords
                H, W = heatmaps.shape[1], heatmaps.shape[2]
                keypoints = coords_2d * np.array([W, H])
                keypoints = keypoints[np.newaxis, ...]
                scores = heatmaps.max(axis=(1, 2))
                scores = scores[np.newaxis, ...]

            pred = InstanceData()
            pred.keypoints = keypoints
            pred.keypoint_scores = scores
            pred.keypoint_3d = pose_3d[np.newaxis, ...]

            preds.append(pred)

        return preds
