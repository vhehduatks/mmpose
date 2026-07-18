"""Task 14.2 — Ours-PT: ray-temporal depth context on the 14d (s12) base
(new module file; no tracked mmpose files are edited — registered via
custom_imports).

The claim under test: does past-frame joint information improve DEPTH
prediction when depth is the sole unknown? Ours-T / Task 12 measured
temporal history on a free-xyz head only; here the base is the end-to-end
pinhole depth cascade (14d: coarse = ray*d1, final = ray*(d1+dd)), and the
pooled ray-history feature f_temp is appended to the stage-2 depth-delta
input. NOTE (framing): 14d carries a ~12 mm ray-lock handicap vs baseline
(FOV ceiling + 41-px uv quantization) — the comparison that matters is
14e vs 14d and vs the controls, NOT vs baseline.

rays_mode selects the arm:
    "plucker"   14e main: floor-frame Plücker ray history (Ours-T shell).
    "direction" ablation: ray directions only, moment/origin term zeroed.
    "repcur"    MANDATORY matched-capacity control: the last (current) step
                repeated over the whole history — same parameters, zero
                temporal information.
    "coords"    MANDATORY coords-only control: camera-frame unit rays from
                raw 2D history, no HMD/device poses at all.

f_temp enters through zero-initialized w1 columns (and the 14d checkpoint
is padded with zero columns on load), so at init every arm reproduces 14d
EXACTLY; masked (warm-up) samples fall back to 14d per-sample. Reuses
KinectEgoposeTemporalDataset / OursTemporalCodec / RayTemporalTransformer /
FreezeStage1Hook from ours_t_modules (imported => registered).

Codebase rules honored: .reshape() only, new files only.
"""

import torch
import torch.nn as nn

from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    RefinementMLP, compute_kinematic_features, soft_argmax_2d)
from mmpose.registry import MODELS

from my_code.custom_config.ours_p_modules import OursPinholeHead
from my_code.custom_config.ours_t_modules import (  # noqa: F401  (registers dataset/codec/hook)
    K_CX, K_CY, K_FX, K_FY, KinectEgoposeTemporalDataset, OursTemporalCodec,
    FreezeStage1Hook, RayTemporalTransformer, plucker_rays_floor)


@MODELS.register_module()
class OursPTHead(OursPinholeHead):
    """s12 pinhole head + f_temp (ray temporal feature) in the stage-2
    depth-delta input."""

    def __init__(self, *args, f_temp_dim: int = 64, history: int = 20,
                 temporal_embed: int = 512, temporal_heads: int = 32,
                 temporal_layers: int = 8, zero_f_temp: bool = False,
                 rays_mode: str = "plucker",
                 refinement_hidden_size: int = 256,
                 refinement_num_stage: int = 1,
                 refinement_dropout: float = 0.5, **kwargs):
        super().__init__(*args,
                         refinement_hidden_size=refinement_hidden_size,
                         refinement_num_stage=refinement_num_stage,
                         refinement_dropout=refinement_dropout, **kwargs)
        assert self.pinhole_mode == "s12", "Ours-PT targets the 14d base"
        assert rays_mode in ("plucker", "direction", "repcur", "coords"), \
            rays_mode
        self.f_temp_dim = f_temp_dim
        self.zero_f_temp = zero_f_temp
        self.rays_mode = rays_mode
        self._f_temp = None

        self.temporal_transformer = RayTemporalTransformer(
            num_keypoints=self.out_channels, history=history,
            embed_dim=temporal_embed, num_heads=temporal_heads,
            num_layers=temporal_layers, out_dim=f_temp_dim)

        # rebuild the depth-delta MLP with room for f_temp; zero-init the
        # new input columns so f_temp is a no-op at init.
        self._base_ref_in = self.refinement_mlp.w1.in_features
        self.refinement_mlp = RefinementMLP(
            input_size=self._base_ref_in + f_temp_dim,
            hidden_size=refinement_hidden_size,
            num_stage=refinement_num_stage,
            p_dropout=refinement_dropout)
        self.refinement_mlp.w_out = nn.Linear(refinement_hidden_size, 1)
        with torch.no_grad():
            self.refinement_mlp.w1.weight[:, self._base_ref_in:].zero_()

        self._register_load_state_dict_pre_hook(self._pad_w1_f_temp)

    def _pad_w1_f_temp(self, state_dict, prefix, *args):
        """Load the 14d checkpoint: pad refinement_mlp.w1 with zero columns
        for the f_temp inputs. (_graft_z_rows runs first and is a no-op on
        the already-depth-form 14d shapes.)"""
        key = prefix + "refinement_mlp.w1.weight"
        w = state_dict.get(key)
        if w is not None and w.shape[1] == self._base_ref_in:
            pad = w.new_zeros(w.shape[0], self.f_temp_dim)
            state_dict[key] = torch.cat([w, pad], dim=1)

    def _compute_f_temp(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        kp2d = torch.cat([l.temporal_kp2d for l in labels]).to(device).float()
        c2w = torch.cat(
            [l.temporal_cam2world for l in labels]).to(device).float()
        m2w = torch.cat(
            [l.temporal_mid2world for l in labels]).to(device).float()
        mask = torch.cat([l.temporal_mask for l in labels]).to(device).float()
        with torch.no_grad():
            if self.rays_mode == "coords":
                u = (kp2d[..., 0] - K_CX) / K_FX
                v = (kp2d[..., 1] - K_CY) / K_FY
                d_cam = torch.stack([u, v, torch.ones_like(u)], dim=-1)
                d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)
                rays = torch.cat([d_cam, torch.zeros_like(d_cam)], dim=-1)
            else:
                rays = plucker_rays_floor(kp2d, c2w, m2w)
                if self.rays_mode == "direction":
                    rays = torch.cat(
                        [rays[..., :3], torch.zeros_like(rays[..., 3:])],
                        dim=-1)
                elif self.rays_mode == "repcur":
                    rays = rays[:, -1:].expand(
                        -1, rays.shape[1], -1, -1)
        f = self.temporal_transformer(rays)
        f = f * mask.reshape(-1, 1)
        if self.zero_f_temp:
            f = f * 0.0
        return f

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        hm = batch_outputs[0] if isinstance(batch_outputs, tuple) \
            else batch_outputs
        self._f_temp = self._compute_f_temp(batch_data_samples, hm.device)
        try:
            return super().decode(batch_outputs, batch_data_samples,
                                  backbone_feat)
        finally:
            self._f_temp = None

    def loss(self, feats, batch_data_samples, train_cfg={}):
        self._f_temp = self._compute_f_temp(
            batch_data_samples, feats[-1].device)
        try:
            return super().loss(feats, batch_data_samples, train_cfg)
        finally:
            self._f_temp = None

    def refine_pinhole(self, coarse_pose, rays, d1, heatmap, backbone_feat,
                       z_latent, hmd_info=None):
        """Parent's s12 stage 2 with f_temp appended per joint."""
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

        f_temp = self._f_temp
        if f_temp is None:
            f_temp = coarse_pose.new_zeros(B, self.f_temp_dim)
        parts.append(f_temp.unsqueeze(1).expand(B, Kj, -1))

        out = self.refinement_mlp(
            torch.cat(parts, dim=-1).reshape(B * Kj, -1))
        dd = out.reshape(B, Kj)
        return rays * (d1 + dd).unsqueeze(-1)
