"""Task 18 — attention in stage 2: joint self-attn + modality cross-attn
(new module file; no tracked mmpose files are edited — registered via
custom_imports). HEADLINE-EXPOSED (user directive): this changes the core
single-frame model; the 64.09/64.36 row may move.

Structural gap: the parent refine() flattens 16 joints into the batch and
runs a shared MLP per joint — no joint<->joint interaction, and ~288 of
the 355 per-joint input dims are global vectors replicated 16x.

18a (this file) adds a standard transformer-decoder block as a GATED
RESIDUAL next to the untouched MLP path:

    Query : 16 joint tokens = [canon_xyz(3), spatial_feat(64)] -> d=64
      -> self-attention (16<->16)
      -> cross-attention over 4 modality tokens [z, pose_ctx, kin, hmd]
      -> FFN -> per-joint delta(3), output layer ZERO-INIT
    delta = mlp_delta + sigmoid(gate) * attn_delta

Zero-init output => init reproduces the headline DIGIT-FOR-DIGIT (GATE
18-0c) regardless of the gate value.

Canonicalization (user directive): the joint xyz entering the attention
is expressed in the gravity-aligned floor frame (FRAME construction from
the MIDDLE pose: mid2floor · inv(mid2world) · cam2world applied to the
ego-cam coarse), removing head pitch/roll so joint<->joint relations are
consistent across samples. Design (i): canonicalize ONLY the attention's
input representation; the delta head and everything else stay in ego-cam
(no inverse transform). Warm-up/uncached samples (mask=0) fall back to
raw ego-cam coordinates.

Data: reuses KinectEgoposeTemporalDataset with history=1 (per-frame
cam2world/mid2world/mask via OursTemporalCodec) — no new dataset code.

attn_mode arms: "full" | "self_only" | "cross_only" | "mlp_matched"
(parameter-matched plain residual MLP control, ~73k params, h=204);
use_hmd_token=False drops the hmd modality token (double-use ablation —
the device pose is already consumed geometrically by the canonicalization).

Codebase rules honored: .reshape() only, new files only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, compute_kinematic_features,
    soft_argmax_2d)
from mmpose.registry import MODELS

from my_code.custom_config.ours_t_modules import (  # noqa: F401  (registers dataset/codec/hook)
    FreezeStage1Hook, KinectEgoposeTemporalDataset, OursTemporalCodec,
    compute_relpose_to_floor, invert_se3)


def hard_local_argmax_2d(heatmaps):
    """Task 19.1(b): hard-argmax cell + value-weighted local mean over its
    3x3 neighbourhood (border-clamped). Removes the global-tail bias that
    drags the T=1 softmax centroid toward the image centre (GATE 19.0:
    median soft-vs-hard displacement 120-660 px; hard is 36-97 px from GT
    2D vs soft's 123-672). Returns [0,1]-normalized coords like
    soft_argmax_2d."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = flat.argmax(dim=-1)
    y0, x0 = idx // W, idx % W
    offs = torch.arange(-1, 2, device=heatmaps.device)
    ys = (y0[..., None] + offs).clamp(0, H - 1)              # (B,K,3)
    xs = (x0[..., None] + offs).clamp(0, W - 1)
    yy = ys[..., :, None].expand(B, K, 3, 3).reshape(B, K, 9)
    xx = xs[..., None, :].expand(B, K, 3, 3).reshape(B, K, 9)
    v = torch.gather(flat, 2, yy * W + xx)
    w = v.clamp_min(0) + 1e-6
    w = w / w.sum(dim=-1, keepdim=True)
    x = (w * xx.to(heatmaps.dtype)).sum(-1) / max(W - 1, 1)
    y = (w * yy.to(heatmaps.dtype)).sum(-1) / max(H - 1, 1)
    return torch.stack([x, y], dim=-1)


class JointAttnBlock(nn.Module):
    """Transformer-decoder-style block over 16 joint tokens."""

    def __init__(self, attn_mode="full", d=64, heads=4, dropout=0.1,
                 spatial_dim=64, z_dim=64, pose_dim=128, kin_dim=64,
                 hmd_dim=32, use_hmd_token=True):
        super().__init__()
        self.attn_mode = attn_mode
        self.use_hmd_token = use_hmd_token
        self.joint_embed = nn.Linear(3 + spatial_dim, d)
        if attn_mode in ("full", "self_only"):
            self.self_attn = nn.MultiheadAttention(
                d, heads, dropout=dropout, batch_first=True)
            self.ln1 = nn.LayerNorm(d)
        if attn_mode in ("full", "cross_only"):
            self.mod_z = nn.Linear(z_dim, d)
            self.mod_pose = nn.Linear(pose_dim, d)
            self.mod_kin = nn.Linear(kin_dim, d)
            if use_hmd_token:
                self.mod_hmd = nn.Linear(hmd_dim, d)
            self.cross_attn = nn.MultiheadAttention(
                d, heads, dropout=dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))
        self.ln3 = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)
        with torch.no_grad():                      # init-identity guarantee
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, canon_xyz, spatial_feat, z, pose_feat, kin_feat,
                hmd_feat):
        q = self.joint_embed(torch.cat([canon_xyz, spatial_feat], dim=-1))
        if self.attn_mode in ("full", "self_only"):
            a, _ = self.self_attn(q, q, q)
            q = self.ln1(q + a)
        if self.attn_mode in ("full", "cross_only"):
            toks = [self.mod_z(z), self.mod_pose(pose_feat),
                    self.mod_kin(kin_feat)]
            if self.use_hmd_token and hmd_feat is not None:
                toks.append(self.mod_hmd(hmd_feat))
            kv = torch.stack(toks, dim=1)                    # (B, 3|4, d)
            a, _ = self.cross_attn(q, kv, kv)
            q = self.ln2(q + a)
        q = self.ln3(q + self.ffn(q))
        return self.out(q)                                   # (B, 16, 3)


class MatchedMLPBlock(nn.Module):
    """Parameter-matched plain residual control: per-joint flat input ->
    h -> 3, zero-init output. h=204 -> 359*204+3 = 73,239 params vs the
    full attention block's ~73,331 (see RESULTS for exact counts)."""

    def __init__(self, in_dim=355, h=204):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(),
                                 nn.Linear(h, 3))
        with torch.no_grad():
            self.net[2].weight.zero_()
            self.net[2].bias.zero_()

    def forward(self, joint_flat):
        return self.net(joint_flat)


@MODELS.register_module()
class OursAttnCascadedHead(CustomEgoposeCascadedRefinementHead_enhanced):

    def __init__(self, *args, attn_mode: str = "full",
                 use_hmd_token: bool = True, attn_dim: int = 64,
                 attn_heads: int = 4, use_canon: bool = True,
                 sample_mode: str = "soft", sample_temp: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert attn_mode in ("full", "self_only", "cross_only",
                             "mlp_matched", "replace"), attn_mode
        # Task 19.1 sampling variants (param-free): 'soft' = parent behavior
        # (T=1 softmax centroid), 'temp' = sharpened softmax at sample_temp,
        # 'hard_local' = hard-argmax + 3x3 value-weighted local mean.
        assert sample_mode in ("soft", "temp", "hard_local"), sample_mode
        self.sample_mode = sample_mode
        self.sample_temp = float(sample_temp)
        self.attn_mode = attn_mode
        # ablation: feed RAW ego-cam joint tokens instead of floor-frame
        # canonicalized ones (attributes canonicalization vs attention).
        self.use_canon = bool(use_canon)
        self._canon = None
        self.attn_gate = nn.Parameter(torch.zeros(16, 1))
        if attn_mode == "mlp_matched":
            self.attn_block = MatchedMLPBlock()
        else:
            # 18b "replace": the decoder block IS stage 2 (full self+cross,
            # no MLP path, replicated globals dropped); retrained, no
            # warm-start guarantee.
            block_mode = "full" if attn_mode == "replace" else attn_mode
            self.attn_block = JointAttnBlock(
                attn_mode=block_mode, d=attn_dim, heads=attn_heads,
                use_hmd_token=use_hmd_token)

    # -- canonical transform from the per-sample device poses --------------
    def _canon_from_samples(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        c2w = torch.cat([l.temporal_cam2world for l in labels]
                        ).to(device).float()[:, 0]           # (B,4,4)
        m2w = torch.cat([l.temporal_mid2world for l in labels]
                        ).to(device).float()                 # (B,4,4)
        mask = torch.cat([l.temporal_mask for l in labels]
                         ).to(device).float()                # (B,)
        T = compute_relpose_to_floor(m2w) @ invert_se3(m2w) @ c2w
        eye = torch.eye(4, device=device).expand_as(T)
        return torch.where(mask.reshape(-1, 1, 1) > 0, T, eye)

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        hm = batch_outputs[0] if isinstance(batch_outputs, tuple) \
            else batch_outputs
        self._canon = self._canon_from_samples(batch_data_samples, hm.device)
        try:
            return super().decode(batch_outputs, batch_data_samples,
                                  backbone_feat)
        finally:
            self._canon = None

    def loss(self, feats, batch_data_samples, train_cfg={}):
        self._canon = self._canon_from_samples(
            batch_data_samples, feats[-1].device)
        try:
            return super().loss(feats, batch_data_samples, train_cfg)
        finally:
            self._canon = None

    # -- stage 2: parent body + gated attention residual --------------------
    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent,
               hmd_info=None):
        B, K = coarse_pose.shape[0], coarse_pose.shape[1]

        hm_det = heatmap.detach()
        if self.sample_mode == "hard_local":
            coords_2d = hard_local_argmax_2d(hm_det)
        else:
            t = self.sample_temp if self.sample_mode == "temp" else 1.0
            coords_2d, _ = soft_argmax_2d(hm_det, temperature=t)
        grid = (coords_2d * 2 - 1).unsqueeze(1)
        sampled = F.grid_sample(backbone_feat, grid, mode="bilinear",
                                align_corners=True, padding_mode="border")
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        spatial_feat = self.spatial_proj(sampled)

        pose_feat = self.pose_encoder_net(coarse_pose.reshape(B, -1))
        kin_feat = self.kin_encoder(compute_kinematic_features(coarse_pose))

        if self.use_hmd_in_refinement and hmd_info is not None:
            hmd_feat = self.hmd_encoder_stage2(hmd_info.to(torch.float32))
            hmd_exp = hmd_feat.unsqueeze(1).expand(B, K, -1)
        else:
            hmd_feat, hmd_exp = None, None

        z_exp = z_latent.unsqueeze(1).expand(B, K, -1)
        pose_exp = pose_feat.unsqueeze(1).expand(B, K, -1)
        kin_exp = kin_feat.unsqueeze(1).expand(B, K, -1)
        parts = [coarse_pose, spatial_feat, z_exp, pose_exp, kin_exp]
        if hmd_exp is not None:
            parts.append(hmd_exp)
        joint_input = torch.cat(parts, dim=-1)

        if self.attn_mode == "replace":
            if self.use_canon and self._canon is not None:
                R, t = self._canon[:, :3, :3], self._canon[:, :3, 3]
                canon = torch.einsum(
                    "bij,bkj->bki", R, coarse_pose) + t[:, None]
            else:
                canon = coarse_pose
            delta = self.attn_block(canon, spatial_feat, z_latent,
                                    pose_feat, kin_feat, hmd_feat)
            return coarse_pose + delta

        delta = self.refinement_mlp(
            joint_input.reshape(B * K, -1)).reshape(B, K, 3)

        # gated attention residual
        if self.attn_mode == "mlp_matched":
            extra = self.attn_block(
                joint_input.reshape(B * K, -1)).reshape(B, K, 3)
        else:
            if self.use_canon and self._canon is not None:
                R, t = self._canon[:, :3, :3], self._canon[:, :3, 3]
                canon = torch.einsum(
                    "bij,bkj->bki", R, coarse_pose) + t[:, None]
            else:
                canon = coarse_pose
            extra = self.attn_block(canon, spatial_feat, z_latent,
                                    pose_feat, kin_feat, hmd_feat)
        delta = delta + torch.sigmoid(self.attn_gate) * extra

        return coarse_pose + delta
