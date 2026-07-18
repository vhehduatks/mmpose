"""Task 10 — Ours-G: gravity-aligned view-direction conditioning (new module
file; no tracked mmpose files are edited — registered via custom_imports).

Single-frame (NOT temporal). Fixes the mechanism Ours-T diagnosed: rays as a
separate token stream through a 64-d bottleneck barely couple to image
features. Instead the missing geometry is injected PIXEL-ALIGNED, at the exact
point stage 2 samples image features: per joint, at its decoded 2D location,

    d_f = R_floor_cam · normalize(K^-1 [u, v, 1]^T)   (gravity-aligned view dir)
    h   = HMD/camera height above floor (scalar)

with R_floor_cam from the CURRENT frame's HMD pose via the same y-up floor
construction as STF / Ours-T (heading normalization keeps yaw invariance).
No history, no kp2d caches, no warm-up. Direction + height together determine
where each joint's ray meets the floor — the gravity/floor prior Task 8.9
isolated (8.9 mm), now available per pixel to the lifting stage.

Components:
- KinectEgoposeGeoDataset: paper loader + per-sample CURRENT-frame device
  poses (frame_export on_device_poses). Missing poses -> geo_mask=0 -> the
  head falls back to the exact single-frame baseline path.
- OursGeoCodec: label mapping for the geo fields.
- OursGeoCascadedHead: cascaded head; per-joint [d_f (3), h (1)] is
  concatenated into the stage-2 refinement input, evaluated at the SAME
  soft-argmax 2D coordinates stage 2 already samples backbone features at.
  The pretrained stage-2 w1 loads into the leading columns and the geo
  columns start at zero, so zero_geo=True (GATE 10b) reproduces the baseline
  EXACTLY. `geo_mode="camera"` skips gravity alignment (control: camera-frame
  direction is a pure pixel encoding); `use_height=False` drops the h term.
- FreezeStage1Hook is reused from ours_t_modules (imported -> registered).

Codebase rules honored: .reshape() only, new files only.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmengine.structures import InstanceData
from mmpose.codecs.custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, RefinementMLP,
    compute_kinematic_features, preprocess_hmd_data_batch, soft_argmax_2d)
from mmpose.registry import DATASETS, KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy

from my_code.custom_config.ours_t_modules import (  # noqa: F401 (registers FreezeStage1Hook)
    K_CX, K_CY, K_FX, K_FY, FreezeStage1Hook, compute_relpose_to_floor,
    invert_se3)

# Original image size the calibrated K is expressed in (EgoImageResize is a
# full-image stretch, so soft-argmax coords in [0,1] map linearly to these).
IMG_W, IMG_H = 1920.0, 1080.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgoposeGeoDataset(KinectEgoposeDataset):
    """Paper loader + CURRENT-frame device poses for the geo conditioning.

    Extra per-sample fields (numpy, packed to gt_instance_labels by
    OursGeoCodec's label mapping):
        geo_cam2world (1,4,4)  left-cam pose, frame_export y-up world
        geo_mid2world (1,4,4)  middle/HMD pose (floor-frame construction)
        geo_mask      (1,)     1.0 iff device poses available for this frame
    """

    def __init__(self, *, frame_export_root: str, **kwargs):
        self.frame_export_root = Path(frame_export_root)
        super().__init__(**kwargs)

    @staticmethod
    def _to44(z):
        T = z["rotations"].shape[0]
        M = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        M[:, :3, :3] = z["rotations"].astype(np.float32)
        M[:, :3, 3] = z["translations"].astype(np.float32)
        return M

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        per_session = {}
        n_masked = 0
        eye = np.eye(4, dtype=np.float32)[None]

        for d in data_list:
            img = d["img_path"]
            m = pat.search(img)
            assert m, img
            fid = int(m.group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            key = (participant, session)
            if key not in per_session:
                act = self.frame_export_root / participant / "actions" / session
                pose_l = act / "on_device_poses" / "egocam_left.npz"
                pose_m = act / "on_device_poses" / "egocam_middle.npz"
                if pose_l.is_file() and pose_m.is_file():
                    per_session[key] = (self._to44(np.load(pose_l)),
                                        self._to44(np.load(pose_m)))
                else:
                    per_session[key] = None

            entry = per_session[key]
            if entry is None or fid >= entry[0].shape[0]:
                d["geo_cam2world"] = eye
                d["geo_mid2world"] = eye
                d["geo_mask"] = np.zeros(1, dtype=np.float32)
                n_masked += 1
            else:
                c2w, m2w = entry
                d["geo_cam2world"] = c2w[fid][None]
                d["geo_mid2world"] = m2w[fid][None]
                d["geo_mask"] = np.ones(1, dtype=np.float32)
        print(f"[KinectEgoposeGeoDataset] {len(data_list)} samples, "
              f"{n_masked} masked (no device poses)")
        return data_list


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

@KEYPOINT_CODECS.register_module()
class OursGeoCodec(Custom_mo2cap2_MSRAHeatmap):
    label_mapping_table = dict(
        Custom_mo2cap2_MSRAHeatmap.label_mapping_table,
        geo_cam2world="geo_cam2world",
        geo_mid2world="geo_mid2world",
        geo_mask="geo_mask",
    )


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class OursGeoCascadedHead(CustomEgoposeCascadedRefinementHead_enhanced):
    """Cascaded head + per-joint gravity-aligned view direction in stage 2.

    The pretrained refinement w1 loads into the first columns; the geo
    columns are zero-initialized, so at init (and whenever the geo feature is
    zeroed by mask or `zero_geo`) the head is EXACTLY the pretrained baseline.
    """

    def __init__(self, *args, zero_geo: bool = False,
                 geo_mode: str = "gravity", use_height: bool = True,
                 refinement_hidden_size: int = 256,
                 refinement_num_stage: int = 1,
                 refinement_dropout: float = 0.5,
                 spatial_feat_dim: int = 64, pose_feat_dim: int = 128,
                 kin_feat_dim: int = 64, **kwargs):
        super().__init__(*args,
                         refinement_hidden_size=refinement_hidden_size,
                         refinement_num_stage=refinement_num_stage,
                         refinement_dropout=refinement_dropout,
                         spatial_feat_dim=spatial_feat_dim,
                         pose_feat_dim=pose_feat_dim,
                         kin_feat_dim=kin_feat_dim, **kwargs)
        assert self.use_refinement, "Ours-G injects into stage 2"
        assert geo_mode in ("gravity", "camera"), geo_mode
        self.zero_geo = zero_geo
        # "camera": ablation control — skip gravity alignment; the direction
        # becomes a pure function of the pixel (no HMD information in the
        # direction term). Height is kept so the control isolates rotation.
        self.geo_mode = geo_mode
        self.use_height = use_height
        self.geo_dim = 3 + (1 if use_height else 0)

        hmd_dim = 32 if self.use_hmd_in_refinement else 0
        self._base_ref_in = (3 + spatial_feat_dim + 64 + pose_feat_dim
                             + kin_feat_dim + hmd_dim)
        self.refinement_mlp = RefinementMLP(
            input_size=self._base_ref_in + self.geo_dim,
            hidden_size=refinement_hidden_size,
            num_stage=refinement_num_stage,
            p_dropout=refinement_dropout)
        with torch.no_grad():
            self.refinement_mlp.w1.weight[:, self._base_ref_in:].zero_()

        self._register_load_state_dict_pre_hook(self._pad_refinement_w1)

    def _pad_refinement_w1(self, state_dict, prefix, *args):
        """Load a baseline checkpoint: pad refinement_mlp.w1 with zero
        columns for the geo inputs."""
        key = prefix + "refinement_mlp.w1.weight"
        w = state_dict.get(key)
        if w is not None and w.shape[1] == self._base_ref_in:
            pad = w.new_zeros(w.shape[0], self.geo_dim)
            state_dict[key] = torch.cat([w, pad], dim=1)

    def _compute_geo_frames(self, batch_data_samples, device):
        """Per-sample rotation-to-conditioning-frame + height + mask.

        Returns (R (B,3,3), h (B,1), mask (B,1)); R maps camera-frame
        directions to the gravity-aligned floor frame ("gravity") or is
        identity ("camera" control).
        """
        labels = [d.gt_instance_labels for d in batch_data_samples]
        c2w = torch.cat([l.geo_cam2world for l in labels]).to(device).float()
        m2w = torch.cat([l.geo_mid2world for l in labels]).to(device).float()
        mask = torch.cat([l.geo_mask for l in labels]).to(device).float()
        with torch.no_grad():
            mid2floor = compute_relpose_to_floor(m2w)
            cam2floor = mid2floor @ invert_se3(m2w) @ c2w
            h = cam2floor[:, 1, 3]
            if self.geo_mode == "gravity":
                R = cam2floor[:, :3, :3]
            else:
                R = torch.eye(3, device=device).unsqueeze(0).expand(
                    c2w.shape[0], 3, 3)
        return R, h.reshape(-1, 1), mask.reshape(-1, 1)

    def _geo_feature(self, coords_2d, geo):
        """Per-joint [d (3), h (1)] at the stage-2 sampling coordinates.

        coords_2d: (B,K,2) soft-argmax coords in [0,1] (full-image resize ->
        linear map to the 1920x1080 px frame K is calibrated in).
        """
        B, Kj = coords_2d.shape[0], coords_2d.shape[1]
        if geo is None:
            return coords_2d.new_zeros(B, Kj, self.geo_dim)
        R, h, mask = geo
        u = coords_2d[..., 0] * IMG_W
        v = coords_2d[..., 1] * IMG_H
        x = (u - K_CX) / K_FX
        y = (v - K_CY) / K_FY
        d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)
        d = torch.einsum("bij,bkj->bki", R, d_cam)
        parts = [d]
        if self.use_height:
            parts.append(h.unsqueeze(1).expand(B, Kj, 1))
        g = torch.cat(parts, dim=-1) * mask.reshape(-1, 1, 1)
        if self.zero_geo:
            g = g * 0.0
        return g

    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent,
               hmd_info=None, geo=None):
        """Stage 2 with the per-joint geo feature appended."""
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
        parts.append(self._geo_feature(coords_2d, geo))

        joint_input = torch.cat(parts, dim=-1)
        delta = self.refinement_mlp(joint_input.reshape(B * Kj, -1))
        return coarse_pose + delta.reshape(B, Kj, 3)

    # decode() and loss() are copied from the parent with the single change
    # that refine() receives the geo frames (the parent hard-codes the call).

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
        batch_3d_keypoints = self.pose_decoder(z_plus_hmd)

        if self.use_auxiliary_decoders:
            generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
            hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)
        else:
            generated_heatmaps = None
            hmd_recons = None

        if self.use_refinement and backbone_feat is not None:
            geo = self._compute_geo_frames(batch_data_samples,
                                           batch_outputs.device)
            coarse_pose = batch_3d_keypoints.reshape(-1, 16, 3)
            output_3d = self.refine(coarse_pose, batch_outputs, backbone_feat,
                                    z, hmd_info=HMD_info, geo=geo)
        else:
            output_3d = batch_3d_keypoints

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
        coarse_pose = self.pose_decoder(z_plus_hmd).reshape(-1, 16, 3)

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

        if self.use_auxiliary_decoders:
            recon_heatmap = self.heatmap_decoder(z_plus_hmd)
            hmd_recon = preprocess_hmd_data_batch(coarse_pose)
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmap, gt_heatmaps, keypoint_weights)
            loss_hmd = self.loss_hmd_module(
                hmd_recon.to(torch.double), HMD_info[:, :9].to(torch.double))
            losses.update(loss_heatmap_recon=loss_heatmap_recon,
                          loss_hmd=loss_hmd)

        geo = self._compute_geo_frames(batch_data_samples, pred_fields.device)
        refined_pose = self.refine(coarse_pose, pred_fields, backbone_feat,
                                   z, hmd_info=HMD_info, geo=geo)
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
