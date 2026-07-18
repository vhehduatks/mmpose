# Copyright (c) OpenMMLab. All rights reserved.
"""
Cascaded Pose Refinement Head with Depth Enhancement

Extends CustomEgoposeCascadedRefinementHead_enhanced by adding depth
grid-sampling in Stage 2. The depth map (provided as input, not predicted)
is processed through a small CNN encoder, then grid-sampled at predicted
2D joint locations to provide per-joint depth features.

Stage 2 input per joint:
  coarse_xyz(3) + spatial(64) + depth(depth_feat_dim) + z(64) +
  pose_ctx(128) + kin(64) + hmd(32) = 355 + depth_feat_dim
"""
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType

from .custom_egopose_cascaded_refinement_head_enhanced import (
    CustomEgoposeCascadedRefinementHead_enhanced,
    soft_argmax_2d, compute_kinematic_features,
)

OptIntSeq = Optional[Sequence[int]]


class DepthEncoder(nn.Module):
    """Small CNN to encode a single-channel depth map into a feature volume.

    Input:  (B, 1, 256, 256)
    Output: (B, out_channels, 64, 64)

    Three conv layers with stride-2 downsample: 256→128→64→64 (last is stride-1).
    """

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 256→128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128→64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # 64→64
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth_map):
        """
        Args:
            depth_map: (B, 1, H, W) float32 normalized depth

        Returns:
            (B, out_channels, H//4, W//4) feature volume
        """
        return self.net(depth_map)


@MODELS.register_module()
class CustomEgoposeCascadedRefinementHead_depth(
        CustomEgoposeCascadedRefinementHead_enhanced):
    """Cascaded head with depth map enhancement in Stage 2.

    Adds a DepthEncoder CNN + grid-sampled depth features to the per-joint
    refinement input. All other functionality inherited from _enhanced.

    Additional Args:
        use_depth (bool): Whether to use depth features. Default: True.
        depth_feat_dim (int): Output dim of depth feature per joint. Default: 16.
        depth_encoder_channels (int): Internal channels of DepthEncoder. Default: 32.
    """

    def __init__(self,
                 *args,
                 use_depth: bool = True,
                 depth_feat_dim: int = 16,
                 depth_encoder_channels: int = 32,
                 **kwargs):

        # Temporarily disable refinement build in parent (we rebuild with new size)
        self._depth_feat_dim = depth_feat_dim
        self._depth_encoder_channels = depth_encoder_channels
        self._use_depth = use_depth

        super().__init__(*args, **kwargs)

        # Build depth modules
        if use_depth and self.use_refinement:
            self.depth_encoder = DepthEncoder(out_channels=depth_encoder_channels)
            self.depth_proj = nn.Sequential(
                nn.Linear(depth_encoder_channels, depth_feat_dim),
                nn.ReLU(inplace=True),
            )

            # Rebuild refinement_mlp with increased input size
            # Parent built it with: 3 + spatial + 64 + pose + kin + hmd_stage2
            # We need to add depth_feat_dim
            from .custom_egopose_cascaded_refinement_head_enhanced import (
                RefinementMLP,
            )
            old_input_size = self.refinement_mlp.w1.in_features
            new_input_size = old_input_size + depth_feat_dim

            # Get refinement params from parent init
            # Infer from existing MLP
            hidden_size = self.refinement_mlp.w1.out_features
            num_stage = len(self.refinement_mlp.stages)
            p_dropout = self.refinement_mlp.stages[0].dropout.p if num_stage > 0 else 0.5

            self.refinement_mlp = RefinementMLP(
                input_size=new_input_size,
                hidden_size=hidden_size,
                num_stage=num_stage,
                p_dropout=p_dropout,
            )

    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent,
               hmd_info=None, depth_feat_map=None):
        """Stage 2 with optional depth features.

        Args:
            coarse_pose: [B, 16, 3]
            heatmap: [B, 16, 47, 47]
            backbone_feat: [B, C, H, W]
            z_latent: [B, 64]
            hmd_info: [B, hmd_info_size]
            depth_feat_map: [B, depth_encoder_channels, H_d, W_d] or None

        Returns:
            refined_pose: [B, 16, 3]
        """
        B = coarse_pose.shape[0]
        K = coarse_pose.shape[1]

        # 1. Per-joint spatial features via grid sampling (from backbone)
        coords_2d, _ = soft_argmax_2d(heatmap.detach())
        grid = coords_2d * 2 - 1  # (B, 16, 2) in [-1, 1]
        grid_4d = grid.unsqueeze(1)  # (B, 1, 16, 2)

        sampled = F.grid_sample(
            backbone_feat, grid_4d,
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, 16, C)
        spatial_feat = self.spatial_proj(sampled)  # (B, 16, 64)

        # 2. Depth features via grid sampling (from depth encoder)
        if self._use_depth and depth_feat_map is not None:
            depth_sampled = F.grid_sample(
                depth_feat_map, grid_4d,
                mode='bilinear', align_corners=True, padding_mode='border'
            )
            depth_sampled = depth_sampled.squeeze(2).permute(0, 2, 1)  # (B, 16, C_d)
            depth_feat = self.depth_proj(depth_sampled)  # (B, 16, depth_feat_dim)
        else:
            depth_feat = None

        # 3. Pose context encoding
        pose_flat = coarse_pose.reshape(B, -1)
        pose_feat = self.pose_encoder_net(pose_flat)

        # 4. Kinematic chain features
        kin_raw = compute_kinematic_features(coarse_pose)
        kin_feat = self.kin_encoder(kin_raw)

        # 5. HMD encoding for Stage 2
        if self.use_hmd_in_refinement and hmd_info is not None:
            hmd_feat = self.hmd_encoder_stage2(hmd_info.to(torch.float32))
            hmd_exp = hmd_feat.unsqueeze(1).expand(B, K, -1)
        else:
            hmd_exp = None

        # 6. Build per-joint input
        z_exp = z_latent.unsqueeze(1).expand(B, K, -1)
        pose_exp = pose_feat.unsqueeze(1).expand(B, K, -1)
        kin_exp = kin_feat.unsqueeze(1).expand(B, K, -1)

        parts = [coarse_pose, spatial_feat]
        if depth_feat is not None:
            parts.append(depth_feat)
        parts.extend([z_exp, pose_exp, kin_exp])
        if hmd_exp is not None:
            parts.append(hmd_exp)
        joint_input = torch.cat(parts, dim=-1)

        # 7. Shared refinement MLP
        joint_flat = joint_input.reshape(B * K, -1)
        delta = self.refinement_mlp(joint_flat)
        delta = delta.reshape(B, K, 3)

        return coarse_pose + delta

    def _get_depth_feat_map(self, batch_data_samples):
        """Extract depth maps from data samples and encode."""
        depth_maps = []
        for d in batch_data_samples:
            if hasattr(d, 'depth_map'):
                depth_maps.append(d.depth_map)
            elif hasattr(d, 'gt_instance_labels') and hasattr(d.gt_instance_labels, 'depth_map'):
                depth_maps.append(d.gt_instance_labels.depth_map)
            else:
                return None

        depth_batch = torch.stack(depth_maps)  # (B, 1, H, W)
        if depth_batch.device != next(self.depth_encoder.parameters()).device:
            depth_batch = depth_batch.to(next(self.depth_encoder.parameters()).device)
        return self.depth_encoder(depth_batch.to(torch.float32))

    def loss(self, feats, batch_data_samples, train_cfg={}):
        """Override loss to pass depth to refine()."""
        backbone_feat = feats[-1]
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

        # Stage 1
        z = self.encoder(pred_fields.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)

        coarse_pose = self.pose_decoder(z_plus_hmd)
        coarse_pose = coarse_pose.reshape(-1, 16, 3)

        # Stage 1 losses
        from mmpose.evaluation.functional import pose_pck_accuracy
        from mmpose.utils.tensor_utils import to_numpy
        from .custom_egopose_cascaded_refinement_head_enhanced import (
            preprocess_hmd_data_batch,
        )

        loss_pose_l2norm = self.loss_pose_l2norm_module(coarse_pose, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(coarse_pose, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(coarse_pose, gt_keypoint_3d)
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses = dict()
        losses.update(loss_pose_l2norm=torch.mean(loss_pose_l2norm))
        losses.update(loss_cosine_similarity=torch.mean(loss_cosine))
        losses.update(loss_limb_length=torch.mean(loss_limb))
        losses.update(loss_kpt=loss_kpt)

        # Auxiliary decoders
        if self.use_auxiliary_decoders:
            recon_heatmap = self.heatmap_decoder(z_plus_hmd)
            hmd_recon = preprocess_hmd_data_batch(coarse_pose)
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmap, gt_heatmaps, keypoint_weights)
            HMD_info_base = HMD_info[:, :9]
            loss_hmd = self.loss_hmd_module(
                hmd_recon.to(torch.double), HMD_info_base.to(torch.double))
            losses.update(loss_heatmap_recon=loss_heatmap_recon)
            losses.update(loss_hmd=loss_hmd)

        # Stage 2 with depth
        if self.use_refinement:
            depth_feat_map = self._get_depth_feat_map(batch_data_samples)
            refined_pose = self.refine(
                coarse_pose, pred_fields, backbone_feat, z,
                hmd_info=HMD_info, depth_feat_map=depth_feat_map)

            loss_refined = self.loss_pose_l2norm_refined_module(refined_pose, gt_keypoint_3d)
            loss_bone = self.loss_bone_length_module(refined_pose, gt_keypoint_3d)
            loss_sym = self.loss_symmetry_module(refined_pose)
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

        return losses

    def predict(self, feats, batch_data_samples, test_cfg={}):
        """Override predict to pass depth to decode."""
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            from mmpose.models.utils.tta import flip_heatmaps
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

        # Get depth feature map
        depth_feat_map = self._get_depth_feat_map(batch_data_samples)

        preds, _ = self.decode(
            batch_heatmaps, batch_data_samples,
            backbone_feat=backbone_feat,
            depth_feat_map=depth_feat_map)

        if test_cfg.get('output_heatmaps', False):
            from mmengine.structures import PixelData
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None,
               depth_feat_map=None):
        """Override decode to pass depth to refine."""
        from mmengine.structures import InstanceData
        from mmpose.utils.tensor_utils import to_numpy
        from .custom_egopose_cascaded_refinement_head_enhanced import (
            preprocess_hmd_data_batch,
        )

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(f'Decoder not set in {self.__class__.__name__}.')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)
            if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                batch_scores, batch_visibility = batch_scores
            else:
                batch_visibility = [None] * len(batch_keypoints)
        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints, batch_scores, batch_visibility = [], [], []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples])

        z = self.encoder(batch_outputs.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)

        batch_3d_keypoints = self.pose_decoder(z_plus_hmd)

        if self.use_auxiliary_decoders:
            generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
            hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)
        else:
            generated_heatmaps = None
            hmd_recons = None

        # Stage 2 with depth
        if self.use_refinement and backbone_feat is not None:
            coarse_pose = batch_3d_keypoints.reshape(-1, 16, 3)
            refined_pose = self.refine(
                coarse_pose, batch_outputs, backbone_feat, z,
                hmd_info=HMD_info, depth_feat_map=depth_feat_map)
            output_3d = refined_pose
        else:
            output_3d = batch_3d_keypoints

        preds = []
        for i, (keypoints, kp3d, scores, visibility) in enumerate(zip(
                batch_keypoints, output_3d, batch_scores, batch_visibility)):
            kp3d = kp3d.unsqueeze(dim=0)
            pred_kwargs = dict(keypoints=keypoints, keypoint_scores=scores,
                               keypoint_3d=kp3d)
            if generated_heatmaps is not None:
                pred_kwargs['generated_heatmap'] = generated_heatmaps[i].unsqueeze(0)
            if hmd_recons is not None:
                pred_kwargs['hmd_recon'] = hmd_recons[i].unsqueeze(0)
            pred = InstanceData(**pred_kwargs)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, output_3d
