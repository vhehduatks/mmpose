"""Task 14 — Ours-P: pinhole-lifting variants of the cascaded head (new
module file; no tracked mmpose files are edited — registered via
custom_imports).

3D_cc = ray(uv, K) * depth with the UNNORMALIZED ray
ray(u,v) = ((u-cx)/fx, (v-cy)/fy, 1), so depth == z (the validated z-depth
convention, 0.05 px median reprojection on GT). uv = soft-argmax of the
stage-1 heatmaps mapped to the 1920x1080 px frame (full-image resize).

pinhole_mode selects where the unprojection lives:
    "s2"    (14a) stage-2 pure pinhole: coarse xyz unchanged; the refinement
            MLP outputs a per-joint depth delta; final = ray*(z_coarse+dd).
    "s2res" (14b) 14a + learned residual: refinement outputs [dd, res_xyz];
            final = ray*(z_coarse+dd) + res. Residual rows zero-initialized
            (starts as pure pinhole); hedge for the FOV ceiling — soft-argmax
            uv cannot leave the image cone, but near-camera joints (head/
            neck) can.
    "s1"    (14c) stage-1 pinhole (FRAME's placement): pose_decoder emits
            per-joint depth; coarse = ray*d1; stage 2 stays the existing
            free xyz-delta regression. Adds intermediate L1 depth
            supervision on d1 vs GT z.
    "s12"   (14d) both: coarse = ray*d1, stage-2 outputs a depth delta,
            final = ray*(d1+dd). Depth L1 on d1.

Checkpoint grafting: output layers whose shape changed are initialized from
the headline checkpoint's z-ROWS (pose_decoder.w2 rows [2::3], refinement
w_out row [2]) via a load pre-hook, so depth starts exactly at the
baseline's learned z and (for s2/s2res) the model starts as the analytic
pinhole-ization of the baseline.

Codebase rules honored: .reshape() only, new files only.
"""

import numpy as np
import torch
import torch.nn as nn

from mmengine.structures import InstanceData
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, compute_kinematic_features,
    preprocess_hmd_data_batch, soft_argmax_2d)
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy

from my_code.custom_config.ours_t_modules import K_CX, K_CY, K_FX, K_FY

IMG_W, IMG_H = 1920.0, 1080.0


def hard_uv_px(heatmaps):
    """Sharp per-joint uv in original px from the heatmaps: hard argmax +
    the standard MSRA quarter-cell shift toward the larger neighbor.

    NOTE: soft_argmax_2d is nearly uniform on these low-contrast heatmaps
    (median 380 px from the argmax — a center-biased attention point). It
    stays in use for grid_sample feature sampling (what the checkpoint was
    trained with), but geometry must use the argmax decode (which is what
    the cached kp2d / analytic pinhole-ization used).
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float()
    x = (idx % W).float()
    xi = x.long().clamp(1, W - 2)
    yi = y.long().clamp(1, H - 2)
    bi = torch.arange(B, device=heatmaps.device).reshape(B, 1).expand(B, K)
    ki = torch.arange(K, device=heatmaps.device).reshape(1, K).expand(B, K)
    dx = (heatmaps[bi, ki, yi, xi + 1] - heatmaps[bi, ki, yi, xi - 1])
    dy = (heatmaps[bi, ki, yi + 1, xi] - heatmaps[bi, ki, yi - 1, xi])
    x = x + 0.25 * torch.sign(dx)
    y = y + 0.25 * torch.sign(dy)
    return torch.stack([x * (IMG_W / W), y * (IMG_H / H)], dim=-1)


def rays_from_uv(uv_px):
    """px uv -> unnormalized pinhole rays (B,K,3) with z=1, so ray * depth
    has z == depth."""
    return torch.stack([(uv_px[..., 0] - K_CX) / K_FX,
                        (uv_px[..., 1] - K_CY) / K_FY,
                        torch.ones_like(uv_px[..., 0])], dim=-1)


class DepthLinearModel(nn.Module):
    """LinearModel with a per-joint scalar (depth) output. Same submodule
    names as LinearModel so all non-final weights load verbatim."""

    def __init__(self, input_size, num_classes=16, linear_size=512,
                 num_stage=1, p_dropout=0.5):
        super().__init__()
        from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
            Linear)
        self.w1 = nn.Linear(input_size, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.linear_stages = nn.ModuleList(
            [Linear(linear_size, p_dropout) for _ in range(num_stage)])
        self.w2 = nn.Linear(linear_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.linear_stages:
            y = stage(y)
        return self.w2(y)                                        # (B,16)


@MODELS.register_module()
class OursPinholeHead(CustomEgoposeCascadedRefinementHead_enhanced):

    def __init__(self, *args, pinhole_mode: str = "s2",
                 depth_loss_weight: float = 1.0,
                 refinement_hidden_size: int = 256,
                 refinement_num_stage: int = 1,
                 refinement_dropout: float = 0.5, **kwargs):
        super().__init__(*args,
                         refinement_hidden_size=refinement_hidden_size,
                         refinement_num_stage=refinement_num_stage,
                         refinement_dropout=refinement_dropout, **kwargs)
        assert pinhole_mode in ("s2", "s2res", "s1", "s12"), pinhole_mode
        assert self.use_refinement
        self.pinhole_mode = pinhole_mode
        self.depth_loss_weight = depth_loss_weight

        if pinhole_mode in ("s1", "s12"):
            self.pose_decoder = DepthLinearModel(
                input_size=self.fused_dim, num_classes=16, linear_size=512,
                num_stage=1, p_dropout=0.3)

        out_dims = {"s2": 1, "s2res": 4, "s1": 3, "s12": 1}[pinhole_mode]
        if out_dims != 3:
            self.refinement_mlp.w_out = nn.Linear(
                refinement_hidden_size, out_dims)

        self._register_load_state_dict_pre_hook(self._graft_z_rows)

    def _graft_z_rows(self, state_dict, prefix, *args):
        """Map the headline checkpoint's xyz output layers onto the depth
        output layers: keep the z rows, zero the residual rows (s2res)."""
        if self.pinhole_mode in ("s1", "s12"):
            for suf in ("weight", "bias"):
                key = prefix + "pose_decoder.w2." + suf
                w = state_dict.get(key)
                if w is not None and w.shape[0] == 48:
                    state_dict[key] = w[2::3].clone()
        if self.pinhole_mode in ("s2", "s12", "s2res"):
            for suf in ("weight", "bias"):
                key = prefix + "refinement_mlp.w_out." + suf
                w = state_dict.get(key)
                if w is not None and w.shape[0] == 3:
                    zrow = w[2:3].clone()
                    if self.pinhole_mode == "s2res":
                        state_dict[key] = torch.cat(
                            [zrow, torch.zeros_like(w)], dim=0)
                    else:
                        state_dict[key] = zrow

    def _coarse_from_heatmap(self, batch_heatmaps, z_plus_hmd):
        """Stage-1 output: xyz (baseline) or ray*depth (s1/s12).
        Returns (coarse_pose (B,16,3), rays (B,16,3), d1 or None)."""
        rays = rays_from_uv(hard_uv_px(batch_heatmaps.detach()))
        if self.pinhole_mode in ("s1", "s12"):
            d1 = self.pose_decoder(z_plus_hmd)                   # (B,16)
            coarse = rays * d1.unsqueeze(-1)
        else:
            d1 = None
            coarse = self.pose_decoder(z_plus_hmd).reshape(-1, 16, 3)
        return coarse, rays, d1

    def refine_pinhole(self, coarse_pose, rays, d1, heatmap, backbone_feat,
                       z_latent, hmd_info=None):
        """Stage 2 with the mode's output parametrization."""
        B, Kj = coarse_pose.shape[0], coarse_pose.shape[1]

        coords_2d, _ = soft_argmax_2d(heatmap.detach())
        grid = (coords_2d * 2 - 1).unsqueeze(1)
        sampled = nn.functional.grid_sample(
            backbone_feat, grid, mode="bilinear", align_corners=True,
            padding_mode="border")
        sampled = sampled.reshape(sampled.shape[0], sampled.shape[1], -1)
        sampled = sampled.permute(0, 2, 1)
        spatial_feat = self.spatial_proj(sampled)

        pose_feat = self.pose_encoder_net(coarse_pose.reshape(B, -1))
        kin_feat = self.kin_encoder(compute_kinematic_features(coarse_pose))

        parts = [coarse_pose, spatial_feat,
                 z_latent.unsqueeze(1).expand(B, Kj, -1),
                 pose_feat.unsqueeze(1).expand(B, Kj, -1),
                 kin_feat.unsqueeze(1).expand(B, Kj, -1)]
        if self.use_hmd_in_refinement and hmd_info is not None:
            hmd_feat = self.hmd_encoder_stage2(hmd_info.to(torch.float32))
            parts.append(hmd_feat.unsqueeze(1).expand(B, Kj, -1))

        out = self.refinement_mlp(
            torch.cat(parts, dim=-1).reshape(B * Kj, -1))

        if self.pinhole_mode == "s1":
            return coarse_pose + out.reshape(B, Kj, 3)
        base_depth = d1 if d1 is not None else coarse_pose[..., 2]
        if self.pinhole_mode == "s2":
            dd = out.reshape(B, Kj)
            return rays * (base_depth + dd).unsqueeze(-1)
        if self.pinhole_mode == "s12":
            dd = out.reshape(B, Kj)
            return rays * (base_depth + dd).unsqueeze(-1)
        # s2res
        out = out.reshape(B, Kj, 4)
        return rays * (base_depth + out[..., 0]).unsqueeze(-1) + out[..., 1:]

    # decode()/loss() are copies of the parent with the reparametrized
    # coarse + refine calls (the parent hard-codes both).

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f"The decoder has not been set in {self.__class__.__name__}.")

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
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)
                batch_keypoints.append(keypoints)

        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples])

        z = self.encoder(batch_outputs.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)
        coarse_pose, rays, d1 = self._coarse_from_heatmap(
            batch_outputs, z_plus_hmd)

        if self.use_auxiliary_decoders:
            generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
            hmd_recons = preprocess_hmd_data_batch(coarse_pose)
        else:
            generated_heatmaps = None
            hmd_recons = None

        if self.use_refinement and backbone_feat is not None:
            output_3d = self.refine_pinhole(
                coarse_pose, rays, d1, batch_outputs, backbone_feat, z,
                hmd_info=HMD_info)
        else:
            output_3d = coarse_pose

        preds = []
        for i, (keypoints, kp3d, scores, visibility) in enumerate(zip(
                batch_keypoints, output_3d, batch_scores, batch_visibility)):
            kp3d = kp3d.unsqueeze(dim=0)
            pred_kwargs = dict(keypoints=keypoints, keypoint_scores=scores,
                               keypoint_3d=kp3d)
            if generated_heatmaps is not None:
                pred_kwargs["generated_heatmap"] = \
                    generated_heatmaps[i].unsqueeze(dim=0)
            if hmd_recons is not None:
                pred_kwargs["hmd_recon"] = hmd_recons[i].unsqueeze(dim=0)
            pred = InstanceData(**pred_kwargs)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, output_3d

    def loss(self, feats, batch_data_samples, train_cfg={}):
        from mmpose.evaluation.functional import pose_pck_accuracy

        backbone_feat = feats[-1]
        pred_fields = self.forward(feats)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples])

        z = self.encoder(pred_fields.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)
        coarse_pose, rays, d1 = self._coarse_from_heatmap(
            pred_fields, z_plus_hmd)

        loss_pose_l2norm = self.loss_pose_l2norm_module(
            coarse_pose, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            coarse_pose, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(coarse_pose, gt_keypoint_3d)
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses = dict(
            loss_pose_l2norm=torch.mean(loss_pose_l2norm),
            loss_cosine_similarity=torch.mean(loss_cosine),
            loss_limb_length=torch.mean(loss_limb),
            loss_kpt=loss_kpt)

        if d1 is not None:
            losses.update(loss_depth=self.depth_loss_weight * torch.mean(
                torch.abs(d1 - gt_keypoint_3d[..., 2])))

        if self.use_auxiliary_decoders:
            recon_heatmap = self.heatmap_decoder(z_plus_hmd)
            hmd_recon = preprocess_hmd_data_batch(coarse_pose)
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmap, gt_heatmaps, keypoint_weights)
            loss_hmd = self.loss_hmd_module(
                hmd_recon.to(torch.double), HMD_info[:, :9].to(torch.double))
            losses.update(loss_heatmap_recon=loss_heatmap_recon,
                          loss_hmd=loss_hmd)

        refined_pose = self.refine_pinhole(
            coarse_pose, rays, d1, pred_fields, backbone_feat, z,
            hmd_info=HMD_info)
        loss_refined = self.loss_pose_l2norm_refined_module(
            refined_pose, gt_keypoint_3d)
        loss_bone = self.loss_bone_length_module(refined_pose, gt_keypoint_3d)
        loss_sym = self.loss_symmetry_module(refined_pose)
        losses.update(loss_pose_l2norm_refined=torch.mean(loss_refined),
                      loss_bone_length=torch.mean(loss_bone),
                      loss_symmetry=torch.mean(loss_sym))

        if train_cfg.get("compute_acc", True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            losses.update(acc_pose=torch.tensor(
                avg_acc, device=gt_heatmaps.device))

        self.hm_iteration += 1
        return losses
