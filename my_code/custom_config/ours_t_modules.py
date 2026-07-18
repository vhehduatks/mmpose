"""Task 9 — Ours-T: ray-based temporal lifting (new module file; no tracked
mmpose files are edited — everything is registered from here via
custom_imports in the ours_t configs).

Core idea: a 2D keypoint cannot be moved to a fixed frame as a point, but it
maps EXACTLY to a world-frame ray using only HMD data. Expressed as Plücker
coordinates in the last-step gravity-aligned floor frame (FRAME's
compute_relpose_to_floor, y-up), body motion appears as smooth ray-direction
changes while head motion moves only the ray origins — the two signals that
raw ego 2D entangles.

Components:
- KinectEgoposeTemporalDataset: paper loader + per-sample stage-1 2D history
  (from the Task 9.1 kp2d caches) and device-pose history (frame_export
  on_device_poses). Warm-up (< history steps) or uncached sessions get
  temporal_mask=0 -> the head falls back to the single-frame path.
- OursTemporalCodec: Custom_mo2cap2_MSRAHeatmap + label mapping for the
  temporal fields (that table is how per-sample arrays reach
  gt_instance_labels).
- OursTemporalCascadedHead: the cascaded head + a ray temporal transformer
  (STF-width encoder) whose pooled last-step token f_temp (64-d) is
  concatenated into the stage-2 per-joint refinement input. The pretrained
  stage-2 w1 loads into the first columns and the f_temp columns start at
  zero, so with f_temp zeroed the model reproduces the baseline EXACTLY
  (GATE 9 sanity (b)); `zero_f_temp=True` forces that for gating.
- FreezeStage1Hook: freezes backbone + all stage-1 head modules (params AND
  BatchNorm running stats) — only the temporal branch + stage 2 train.

Codebase rules honored: .reshape() only, new files only.
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from mmengine.hooks import Hook
from mmengine.structures import InstanceData
from mmpose.codecs.custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, RefinementMLP,
    compute_kinematic_features, preprocess_hmd_data_batch, soft_argmax_2d)
from mmpose.registry import DATASETS, HOOKS, KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy

# Calibrated constant pinhole K (Task 7.3; identical across sessions),
# pixels @ 1920x1080 — skeleton_2d IS its projection of skeleton_3d.
K_FX, K_FY, K_CX, K_CY = 1123.768, 1123.214, 965.900, 540.079


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgoposeTemporalDataset(KinectEgoposeDataset):
    """Paper loader + per-sample temporal history for the ray branch.

    Extra per-sample fields (all numpy, packed to gt_instance_labels by
    OursTemporalCodec's label mapping):
        temporal_kp2d      (1,H,16,2) stage-1 2D history, px @1920x1080
        temporal_cam2world (1,H,4,4)  left-cam pose history (frame_export
                                      y-up world, annotation grid)
        temporal_mid2world (1,4,4)    LAST-step middle/HMD pose (floor frame)
        temporal_mask      (1,)       1.0 iff full history + caches available
    """

    def __init__(self, *, frame_export_root: str,
                 kp2d_cache_name: str = "kp2d_ours_pilot",
                 history: int = 20, **kwargs):
        self.frame_export_root = Path(frame_export_root)
        self.kp2d_cache_name = kp2d_cache_name
        self.history = int(history)
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
        H = self.history
        zero_kp = np.zeros((1, H, 16, 2), dtype=np.float32)
        zero_tf = np.tile(np.eye(4, dtype=np.float32), (1, H, 1, 1))
        zero_mid = np.eye(4, dtype=np.float32)[None]

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
                kp_npz = act / "cache" / self.kp2d_cache_name / "joints_2D.npz"
                pose_l = act / "on_device_poses" / "egocam_left.npz"
                pose_m = act / "on_device_poses" / "egocam_middle.npz"
                if kp_npz.is_file() and pose_l.is_file() and pose_m.is_file():
                    per_session[key] = (
                        np.load(kp_npz)["kp2d"].astype(np.float32),
                        self._to44(np.load(pose_l)),
                        self._to44(np.load(pose_m)),
                    )
                else:
                    per_session[key] = None

            entry = per_session[key]
            t_hi = fid
            if entry is None or t_hi >= entry[0].shape[0] or t_hi - H + 1 < 0:
                d["temporal_kp2d"] = zero_kp
                d["temporal_cam2world"] = zero_tf
                d["temporal_mid2world"] = zero_mid
                d["temporal_mask"] = np.zeros(1, dtype=np.float32)
                n_masked += 1
            else:
                kp2d, c2w, m2w = entry
                sl = slice(t_hi - H + 1, t_hi + 1)
                d["temporal_kp2d"] = kp2d[sl][None]
                d["temporal_cam2world"] = c2w[sl][None]
                d["temporal_mid2world"] = m2w[t_hi][None]
                d["temporal_mask"] = np.ones(1, dtype=np.float32)
        print(f"[KinectEgoposeTemporalDataset] {len(data_list)} samples, "
              f"{n_masked} masked (warm-up / uncached), history={H}, "
              f"cache={self.kp2d_cache_name}")
        return data_list


# ---------------------------------------------------------------------------
# Codec (adds the temporal fields to the gt_instance_labels mapping)
# ---------------------------------------------------------------------------

@KEYPOINT_CODECS.register_module()
class OursTemporalCodec(Custom_mo2cap2_MSRAHeatmap):
    label_mapping_table = dict(
        Custom_mo2cap2_MSRAHeatmap.label_mapping_table,
        temporal_kp2d="temporal_kp2d",
        temporal_cam2world="temporal_cam2world",
        temporal_mid2world="temporal_mid2world",
        temporal_mask="temporal_mask",
    )


# ---------------------------------------------------------------------------
# Geometry (ported from framevision.geometry, y-up world)
# ---------------------------------------------------------------------------

def invert_se3(m):
    R = m[..., :3, :3]
    t = m[..., :3, 3]
    inv = torch.zeros_like(m)
    Rt = R.transpose(-1, -2)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3] = (-Rt @ t.unsqueeze(-1)).squeeze(-1)
    inv[..., 3, 3] = 1
    return inv


def compute_relpose_to_floor(pose, align_z_to="x", y_offset=-0.75):
    """FRAME's floor-frame construction (y-up), torch, batched (..., 4, 4)."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[align_z_to]
    axis = pose[..., :3, axis_idx].clone()
    axis[..., 1] = 0
    new_z = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
    new_y = torch.zeros_like(new_z)
    new_y[..., 1] = 1
    new_x = torch.cross(new_y, new_z, dim=-1)
    trans = pose[..., :3, 3].clone()
    trans[..., 1] = 0
    T = torch.zeros_like(pose)
    T[..., :3, 0] = new_x
    T[..., :3, 1] = new_y
    T[..., :3, 2] = new_z
    T[..., :3, 3] = trans
    T[..., 3, 3] = 1
    rel = invert_se3(T) @ pose
    rel[..., 1, 3] += y_offset
    return rel


def plucker_rays_floor(kp2d, cam2world, mid2world_last):
    """2D history -> Plücker rays in the last-step floor frame.

    Args:
        kp2d: (B,H,16,2) px @1920x1080
        cam2world: (B,H,4,4)
        mid2world_last: (B,4,4)

    Returns:
        rays (B,H,16,6): [d_f (unit), m_f = c_f x d_f]
    """
    u = (kp2d[..., 0] - K_CX) / K_FX
    v = (kp2d[..., 1] - K_CY) / K_FY
    dir_cam = torch.stack([u, v, torch.ones_like(u)], dim=-1)
    dir_cam = dir_cam / dir_cam.norm(dim=-1, keepdim=True)          # (B,H,16,3)

    R = cam2world[..., :3, :3]                                       # (B,H,3,3)
    d_w = torch.einsum("bhij,bhkj->bhki", R, dir_cam)                # (B,H,16,3)
    c_w = cam2world[..., :3, 3]                                      # (B,H,3)

    mid2floor = compute_relpose_to_floor(mid2world_last)             # (B,4,4)
    w2f = mid2floor @ invert_se3(mid2world_last)                     # (B,4,4)
    Rf, tf = w2f[:, :3, :3], w2f[:, :3, 3]
    d_f = torch.einsum("bij,bhkj->bhki", Rf, d_w)
    c_f = torch.einsum("bij,bhj->bhi", Rf, c_w) + tf[:, None]
    m_f = torch.cross(c_f[:, :, None].expand_as(d_f), d_f, dim=-1)
    return torch.cat([d_f, m_f], dim=-1)


# ---------------------------------------------------------------------------
# Temporal transformer (mirrors the STF encoder: embed 512)
# ---------------------------------------------------------------------------

class RayTemporalTransformer(nn.Module):
    def __init__(self, num_keypoints=16, history=20, embed_dim=512,
                 num_heads=32, num_layers=8, dropout=0.1, out_dim=64):
        super().__init__()
        self.embedding = nn.Linear(num_keypoints * 6, embed_dim)
        pe = torch.zeros(history, embed_dim)
        pos = torch.arange(history, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32)
                        * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, rays):
        B, H, J, C = rays.shape
        x = self.embedding(rays.reshape(B, H, J * C))
        x = x + self.pos_enc[:, :H]
        x = self.encoder(x)
        return self.out_proj(x[:, -1])                               # (B,out_dim)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class OursTemporalCascadedHead(CustomEgoposeCascadedRefinementHead_enhanced):
    """Cascaded head + f_temp (ray temporal feature) in the stage-2 input.

    The pretrained refinement w1 loads into the first columns; the f_temp
    columns are zero-initialized, so at init (and whenever f_temp is zeroed
    by mask or `zero_f_temp`) the head is EXACTLY the pretrained baseline.
    """

    def __init__(self, *args, f_temp_dim: int = 64, history: int = 20,
                 temporal_layers: int = 8, temporal_heads: int = 32,
                 temporal_embed: int = 512, zero_f_temp: bool = False,
                 rays_mode: str = "plucker",
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
        assert self.use_refinement, "Ours-T injects into stage 2"
        assert rays_mode in ("plucker", "direction"), rays_mode
        self.f_temp_dim = f_temp_dim
        self.zero_f_temp = zero_f_temp
        # "direction": secondary ablation — de-rotated 2D only (floor-frame
        # ray directions, moment/origin term zeroed) to isolate what the
        # camera-translation information adds.
        self.rays_mode = rays_mode

        self.temporal_transformer = RayTemporalTransformer(
            num_keypoints=self.out_channels, history=history,
            embed_dim=temporal_embed, num_heads=temporal_heads,
            num_layers=temporal_layers, out_dim=f_temp_dim)

        # rebuild the refinement MLP with room for f_temp; zero-init the new
        # input columns so f_temp=0 reproduces the baseline exactly.
        hmd_dim = 32 if self.use_hmd_in_refinement else 0
        self._base_ref_in = (3 + spatial_feat_dim + 64 + pose_feat_dim
                             + kin_feat_dim + hmd_dim)
        self.refinement_mlp = RefinementMLP(
            input_size=self._base_ref_in + f_temp_dim,
            hidden_size=refinement_hidden_size,
            num_stage=refinement_num_stage,
            p_dropout=refinement_dropout)
        with torch.no_grad():
            self.refinement_mlp.w1.weight[:, self._base_ref_in:].zero_()

        self._register_load_state_dict_pre_hook(self._pad_refinement_w1)

    def _pad_refinement_w1(self, state_dict, prefix, *args):
        """Load a baseline checkpoint: pad refinement_mlp.w1 with zero
        columns for the f_temp inputs."""
        key = prefix + "refinement_mlp.w1.weight"
        w = state_dict.get(key)
        if w is not None and w.shape[1] == self._base_ref_in:
            pad = w.new_zeros(w.shape[0], self.f_temp_dim)
            state_dict[key] = torch.cat([w, pad], dim=1)

    def _compute_f_temp(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        kp2d = torch.cat([l.temporal_kp2d for l in labels]).to(device).float()
        c2w = torch.cat([l.temporal_cam2world for l in labels]).to(device).float()
        m2w = torch.cat([l.temporal_mid2world for l in labels]).to(device).float()
        mask = torch.cat([l.temporal_mask for l in labels]).to(device).float()
        with torch.no_grad():
            rays = plucker_rays_floor(kp2d, c2w, m2w)
            if self.rays_mode == "direction":
                rays = torch.cat([rays[..., :3],
                                  torch.zeros_like(rays[..., 3:])], dim=-1)
        f = self.temporal_transformer(rays)
        f = f * mask.reshape(-1, 1)
        if self.zero_f_temp:
            f = f * 0.0
        return f

    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent,
               hmd_info=None, f_temp=None):
        """Stage 2 with the f_temp feature appended per joint."""
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
        if f_temp is None:
            f_temp = coarse_pose.new_zeros(B, self.f_temp_dim)
        parts.append(f_temp.unsqueeze(1).expand(B, Kj, -1))

        joint_input = torch.cat(parts, dim=-1)
        delta = self.refinement_mlp(joint_input.reshape(B * Kj, -1))
        return coarse_pose + delta.reshape(B, Kj, 3)

    # decode() and loss() are copied from the parent with the single change
    # that refine() receives f_temp (the parent hard-codes the call).

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

        if self.use_refinement and backbone_feat is not None:
            f_temp = self._compute_f_temp(batch_data_samples,
                                          batch_outputs.device)
            coarse_pose = batch_3d_keypoints.reshape(-1, 16, 3)
            output_3d = self.refine(coarse_pose, batch_outputs, backbone_feat,
                                    z, hmd_info=HMD_info, f_temp=f_temp)
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

        f_temp = self._compute_f_temp(batch_data_samples, pred_fields.device)
        refined_pose = self.refine(coarse_pose, pred_fields, backbone_feat,
                                   z, hmd_info=HMD_info, f_temp=f_temp)
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


# ---------------------------------------------------------------------------
# Freeze hook
# ---------------------------------------------------------------------------

STAGE1_HEAD_MODULES = ("deconv_layers", "add_deconv_layers", "conv_layers",
                       "final_layer", "encoder", "hmd_linear", "pose_decoder",
                       "heatmap_decoder")


@HOOKS.register_module()
class FreezeStage1Hook(Hook):
    """Freeze backbone + stage-1 head modules: no grads AND no BatchNorm
    running-stat updates (re-applied every epoch since the loop calls
    model.train())."""

    def _frozen_modules(self, runner):
        model = runner.model
        model = model.module if hasattr(model, "module") else model
        mods = [model.backbone]
        for name in STAGE1_HEAD_MODULES:
            m = getattr(model.head, name, None)
            if isinstance(m, nn.Module):
                mods.append(m)
        return mods

    def before_train(self, runner):
        n = 0
        for m in self._frozen_modules(runner):
            m.requires_grad_(False)
            n += sum(p.numel() for p in m.parameters())
        runner.logger.info(f"[FreezeStage1Hook] froze {n / 1e6:.1f}M params")

    def before_train_epoch(self, runner):
        for m in self._frozen_modules(runner):
            m.eval()
