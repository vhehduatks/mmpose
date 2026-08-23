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

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.hooks import Hook

from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, compute_kinematic_features,
    soft_argmax_2d)
from mmpose.registry import DATASETS, HOOKS, KEYPOINT_CODECS, MODELS

from my_code.custom_config.ours_t_modules import (  # noqa: F401  (registers dataset/codec/hook)
    FreezeStage1Hook, KinectEgoposeTemporalDataset, OursTemporalCodec,
    compute_relpose_to_floor, invert_se3)


# ---------------------------------------------------------------------------
# Task 21.2 — per-frame device sensor positions in the labels
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgoposeSensorDataset(KinectEgoposeTemporalDataset):
    """Temporal dataset + per-sample raw sensor world positions.

    Adds `sensor_world` (1,9) float32 = [ctrl_left, ctrl_right, hmd] xyz in
    the y-up FRAME world, read from the per-session sensors_v2.npz built by
    ours/frame_adapt/gen_sensors_v2.py (controllers bridged C = M @ S, HMD
    from egocam_middle translations; cross-validated in GATE 17.1a).
    Missing session/frame -> zeros (the zero-init sensor path is a no-op).
    """

    def __init__(self, *, sensors_file: str = "sensors_v2.npz", **kwargs):
        self.sensors_file = sensors_file
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        per_session = {}
        n_missing = 0
        for d in data_list:
            img = d["img_path"]
            fid = int(pat.search(img).group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            key = (participant, session)
            if key not in per_session:
                p = (Path(self.frame_export_root) / participant / "actions"
                     / session / self.sensors_file)
                per_session[key] = (np.load(p)["sensors"].astype(np.float32)
                                    if p.is_file() else None)
            arr = per_session[key]
            if arr is not None and fid < len(arr):
                d["sensor_world"] = arr[fid].reshape(1, 9)
            else:
                d["sensor_world"] = np.zeros((1, 9), dtype=np.float32)
                n_missing += 1
        print(f"[KinectEgoposeSensorDataset] sensors_file={self.sensors_file}, "
              f"{n_missing} samples without sensors")
        return data_list


@KEYPOINT_CODECS.register_module()
class OursSensorCodec(OursTemporalCodec):
    label_mapping_table = dict(
        OursTemporalCodec.label_mapping_table,
        sensor_world="sensor_world",
    )


@HOOKS.register_module()
class FreezeLiftPathHook(Hook):
    """Task 23.1: retrain the LIFTER (encoder + pose_decoder + aux) on
    rotated heatmaps — freeze backbone AND the heatmap-production path
    (deconv/add_deconv/conv/final_layer) so the heatmaps the rotation acts
    on stay fixed."""

    NAMES = ("deconv_layers", "add_deconv_layers", "conv_layers",
             "final_layer")

    def _mods(self, runner):
        model = runner.model
        model = model.module if hasattr(model, "module") else model
        mods = [model.backbone]
        for n in self.NAMES:
            m = getattr(model.head, n, None)
            if isinstance(m, nn.Module):
                mods.append(m)
        return mods

    def before_train(self, runner):
        n = 0
        for m in self._mods(runner):
            m.requires_grad_(False)
            n += sum(p.numel() for p in m.parameters())
        runner.logger.info(f"[FreezeLiftPathHook] froze {n / 1e6:.1f}M params")

    def before_train_epoch(self, runner):
        for m in self._mods(runner):
            m.eval()


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Task 21.2 arm B: stage-1 inputs change (use_hmd=False), so stage 1
    must retrain — freeze ONLY the backbone (grads + BN stats)."""

    def _mods(self, runner):
        model = runner.model
        model = model.module if hasattr(model, "module") else model
        return [model.backbone]

    def before_train(self, runner):
        n = 0
        for m in self._mods(runner):
            m.requires_grad_(False)
            n += sum(p.numel() for p in m.parameters())
        runner.logger.info(f"[FreezeBackboneHook] froze {n / 1e6:.1f}M params")

    def before_train_epoch(self, runner):
        for m in self._mods(runner):
            m.eval()


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
                 hmd_dim=32, use_hmd_token=True, pe_mode="off",
                 sensor_mode="off", sens_subset="all",
                 spatial_in_query=True):
        super().__init__()
        self.attn_mode = attn_mode
        self.use_hmd_token = use_hmd_token
        # Task 21.5 E1-nospat: query = canon_xyz + PE only (no welded
        # appearance anywhere) — measures the pure worth of welding.
        self.spatial_in_query = bool(spatial_in_query)
        # Task 21.2: 3 sensor K/V tokens via a SEPARATE cross-attention
        # whose out_proj is zero-init -> contribution exactly 0 at init
        # (digit-identical warm start). 'floor' = floor-frame xyz (the
        # claim); 'baked' = the existing relative hmd_info parametrization
        # re-tokenized (representation control, same params/tokens);
        # 'const' = learned constant values (capacity control).
        assert sensor_mode in ("off", "floor", "baked", "const"), sensor_mode
        # sens_subset: which floor tokens enter the K/V ('all' | 'ctrl' |
        # 'hmd') — the Task 21.2 attribution ablation. Token order is
        # [ctrl_L, ctrl_R, hmd].
        assert sens_subset in ("all", "ctrl", "hmd"), sens_subset
        self.sensor_mode = sensor_mode
        self.sens_subset = sens_subset
        self._sens_idx = {"all": [0, 1, 2], "ctrl": [0, 1],
                          "hmd": [2]}[sens_subset]
        if sensor_mode != "off":
            self.sens_embed = nn.Linear(3, d)
            self.sens_type = nn.Parameter(torch.zeros(3, d))
            self.sens_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                   batch_first=True)
            with torch.no_grad():
                self.sens_attn.out_proj.weight.zero_()
                self.sens_attn.out_proj.bias.zero_()
            if sensor_mode == "const":
                self.sens_const = nn.Parameter(torch.zeros(3, 3))
        # Task 21.1: learnable joint positional embedding. 'fixed' = the
        # claim (fixed per-joint identity, zero-init => init == t01);
        # 'shuffled' = the control — SAME parameter, fresh random
        # permutation every forward, destroying only the fixed-identity
        # function while holding capacity.
        assert pe_mode in ("off", "fixed", "shuffled"), pe_mode
        self.pe_mode = pe_mode
        if pe_mode != "off":
            self.joint_pe = nn.Parameter(torch.zeros(16, d))
        self.joint_embed = nn.Linear(
            (3 + spatial_dim) if self.spatial_in_query else 3, d)
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
                hmd_feat, sens_toks=None):
        q = self.joint_embed(
            torch.cat([canon_xyz, spatial_feat], dim=-1)
            if self.spatial_in_query else canon_xyz)
        if self.pe_mode == "fixed":
            q = q + self.joint_pe
        elif self.pe_mode == "shuffled":
            q = q + self.joint_pe[torch.randperm(16, device=q.device)]
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
        if self.sensor_mode != "off":
            if self.sensor_mode == "const":
                sens_toks = self.sens_const.unsqueeze(0).expand(
                    q.shape[0], -1, -1)
            idx = self._sens_idx
            kv_s = self.sens_embed(sens_toks[:, idx]) \
                + self.sens_type[idx]                            # (B,n,d)
            a2, _ = self.sens_attn(q, kv_s, kv_s)
            q = q + a2                        # zero-init out_proj: 0 at init
        q = self.ln3(q + self.ffn(q))
        return self.out(q)                                   # (B, 16, 3)


class FactoredAttnBlock(nn.Module):
    """Task 21.3 — the user's intended routing topology.

    Pure-geometry query: joint token = embed(canon_xyz) + joint PE (no
    welded appearance) -> self-attn on kinematics alone. Image evidence is
    CONSULTED, not welded: cross-attn K/V = 16 per-joint spatial_feat
    tokens (each carrying its joint PE, so an occluded joint can attend
    DIRECTLY to a neighbour's image evidence) + [z, pose, kin] globals +
    3 floor-frame sensor tokens = 22 keys — the first many-keys regime
    for cross-attn in this program (Task 17's null was at 3 keys).
    Token-type embedding separates spatial/global/sensor keys (17.1
    lesson). Retrained 18b-style (query embed changes shape 67->3, so no
    zero-init scheme reproduces pesens at init; precedent: 18b replace).
    """

    def __init__(self, d=64, heads=4, dropout=0.1, spatial_dim=64,
                 z_dim=64, pose_dim=128, kin_dim=64):
        super().__init__()
        self.joint_embed = nn.Linear(3, d)
        self.joint_pe = nn.Parameter(torch.zeros(16, d))
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                               batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.spat_embed = nn.Linear(spatial_dim, d)
        self.mod_z = nn.Linear(z_dim, d)
        self.mod_pose = nn.Linear(pose_dim, d)
        self.mod_kin = nn.Linear(kin_dim, d)
        self.sens_embed = nn.Linear(3, d)
        self.sens_type = nn.Parameter(torch.zeros(3, d))
        self.type_emb = nn.Parameter(torch.zeros(3, d))  # spatial/glob/sens
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))
        self.ln3 = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, canon_xyz, spatial_feat, z, pose_feat, kin_feat,
                hmd_feat=None, sens_toks=None):
        q = self.joint_embed(canon_xyz) + self.joint_pe
        a, _ = self.self_attn(q, q, q)
        q = self.ln1(q + a)
        kv_sp = self.spat_embed(spatial_feat) + self.joint_pe \
            + self.type_emb[0]                                # (B,16,d)
        kv_gl = torch.stack([self.mod_z(z), self.mod_pose(pose_feat),
                             self.mod_kin(kin_feat)], dim=1) \
            + self.type_emb[1]                                # (B,3,d)
        kv_se = self.sens_embed(sens_toks) + self.sens_type \
            + self.type_emb[2]                                # (B,3,d)
        kv = torch.cat([kv_sp, kv_gl, kv_se], dim=1)          # (B,22,d)
        a, _ = self.cross_attn(q, kv, kv)
        q = self.ln2(q + a)
        q = self.ln3(q + self.ffn(q))
        return self.out(q)


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

    def __init__(self, *args, frame_joints=(0, 4, 7),
                 frame_loss_weight: float = 1.0, attn_mode: str = "full",
                 use_hmd_token: bool = True, attn_dim: int = 64,
                 attn_heads: int = 4, use_canon: bool = True,
                 sample_mode: str = "soft", sample_temp: float = 0.1,
                 ms_mode: str = "off", pe_mode: str = "off",
                 sensor_mode: str = "off", sens_subset: str = "all",
                 sensor_frame: str = "floor", spatial_in_query: bool = True,
                 film_mode: str = "off",
                 lift_roll_align: bool = False, lift_roll_zero: bool = False,
                 canon_mode: str = "floor",
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Task 38 P3: 'floor' = historical w2f@c2w canon (needs temporal
        # geometry labels); 'body_axis' = calibration-free pose-derived frame
        # (root->pelvis "down" + toe floor level, computed from the detached
        # coarse pose — no extrinsics, works on both datasets).
        # Task 39: 'body_axis_gt' = same body-axis frame but computed from GT
        # keypoint3d (oracle, prediction-independent + noise-free — the
        # ceiling for any allowed frame source under the m2w/w2f/c2w ban).
        # 'frame_head' = Step 2, deployable: MLP z->(g_hat,d) trained by the
        # soft GBH height-consistency loss; canon with the detached frame.
        assert canon_mode in ("floor", "body_axis", "body_axis_gt",
                              "frame_head"), canon_mode
        self.canon_mode = canon_mode
        self._gt_kp3d = None
        self._frame_loss = None
        self._fh = None
        if canon_mode == "frame_head":
            self.frame_head = nn.Sequential(
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
            with torch.no_grad():
                self.frame_head[2].weight.normal_(std=0.01)
                self.frame_head[2].bias.zero_()
            # constant per-point offsets (HMD-above-head / controller-in-hand)
            self.frame_bias = nn.Parameter(torch.zeros(3))
        self.frame_joints = tuple(frame_joints)
        self.frame_loss_weight = float(frame_loss_weight)
        # Task 39-D 'canon_mlp': the 2x2 completion cell — the PLAIN GBH MLP
        # stage-2, with only the 3-d coarse-coord token channel replaced by
        # canonicalized coords (no attention block at all).
        assert attn_mode in ("full", "self_only", "cross_only",
                             "mlp_matched", "replace", "replace_self",
                             "factored", "canon_mlp"), attn_mode
        # Task 20.1 multi-scale spatial feature: 'off' = parent 8x8 sample;
        # 'fused' = 8x8 (2048) + per-joint 47x47 deconv sample (256) -> proj;
        # 'fused_ctrl' = EXACT param match, the 256-d slot filled with the
        # globally-pooled deconv map (same tensor, no per-joint location
        # info) -> isolates per-joint hi-res sampling; 'hi_only' = 47x47
        # sample alone (fewer params).
        assert ms_mode in ("off", "fused", "fused_ctrl", "hi_only"), ms_mode
        self.ms_mode = ms_mode
        self._hi_feat = None
        if ms_mode != "off":
            sf_dim = self.spatial_proj[0].out_features
            hi_ch = 256                        # add_deconv_layers output
            in_dim = hi_ch if ms_mode == "hi_only" \
                else self.spatial_proj[0].in_features + hi_ch
            self.spatial_proj_ms = nn.Sequential(
                nn.Linear(in_dim, sf_dim), nn.ReLU(inplace=True))
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
        if attn_mode == "canon_mlp":
            self.attn_block = None          # plain parent MLP, canon coords
        elif attn_mode == "mlp_matched":
            self.attn_block = MatchedMLPBlock()
        elif attn_mode == "factored":
            self.attn_block = FactoredAttnBlock(d=attn_dim, heads=attn_heads)
        else:
            # 18b "replace": the decoder block IS stage 2 (full self+cross,
            # no MLP path, replicated globals dropped); retrained, no
            # warm-start guarantee.
            block_mode = {"replace": "full",
                          "replace_self": "self_only"}.get(attn_mode,
                                                           attn_mode)
            self.attn_block = JointAttnBlock(
                attn_mode=block_mode, d=attn_dim, heads=attn_heads,
                use_hmd_token=use_hmd_token, pe_mode=pe_mode,
                sensor_mode=sensor_mode, sens_subset=sens_subset,
                spatial_in_query=spatial_in_query)
        # Task 21.5 E2: sensor tokens expressed in 'floor' (canonical) or
        # 'egocam' (shared metric frame only — inv(cam2world) @ p_world).
        assert sensor_frame in ("floor", "egocam"), sensor_frame
        self.sensor_frame = sensor_frame
        self.sensor_mode = sensor_mode
        self._sens = None
        self._w2f = None
        # Task 22a: FiLM-condition spatial_feat on measured ego-cam gravity
        # g_ec = R(cam2world)^T @ [0,1,0] (rotation-only). 'const' control:
        # same MLPs fed a learned constant (capacity without gravity).
        # Zero-init last layers => gamma=1, beta=0 => init == base
        # digit-for-digit.
        assert film_mode in ("off", "gravity", "const"), film_mode
        self.film_mode = film_mode
        self._gvec = None
        # Task 23.1: rotate the ENCODER'S heatmap input by -roll (gravity
        # image-plane angle) before lifting; heatmap losses / codec / refine
        # still see the raw heatmap. lift_roll_zero=True is the identity-
        # roll CONTROL: same wrap, same interpolation blur, angle forced 0.
        # The wrap patches the encoder INSTANCE forward so checkpoint keys
        # are unchanged (warm start intact).
        self.lift_roll_align = bool(lift_roll_align)
        self.lift_roll_zero = bool(lift_roll_zero)
        self._roll = None
        if self.lift_roll_align:
            enc = self.encoder
            orig_fwd = enc.forward

            def _rot_hm(hm, roll):
                B = hm.shape[0]
                c, si = torch.cos(-roll), torch.sin(-roll)
                theta = torch.zeros(B, 2, 3, device=hm.device,
                                    dtype=hm.dtype)
                theta[:, 0, 0], theta[:, 0, 1] = c, -si
                theta[:, 1, 0], theta[:, 1, 1] = si, c
                grid = F.affine_grid(theta, list(hm.shape),
                                     align_corners=True)
                return F.grid_sample(hm, grid, mode="bilinear",
                                     align_corners=True,
                                     padding_mode="zeros")

            def fwd(hm, hmd=None):
                r = self._roll
                if r is not None:
                    hm = _rot_hm(hm, r.to(hm.dtype))
                return orig_fwd(hm) if hmd is None else orig_fwd(hm, hmd)

            enc.forward = fwd
        if film_mode != "off":
            def _film_mlp():
                m = nn.Sequential(nn.Linear(3, 32), nn.GELU(),
                                  nn.Linear(32, 64))
                with torch.no_grad():
                    m[2].weight.zero_()
                    m[2].bias.zero_()
                return m
            self.film_gamma = _film_mlp()
            self.film_beta = _film_mlp()
            if film_mode == "const":
                self.film_in = nn.Parameter(torch.zeros(3))

    # Task 39: canon dispatch — one place deciding the frame source.
    def _apply_canon(self, coarse_pose):
        if self.canon_mode == "body_axis":
            return self._body_axis_canon(coarse_pose)
        if self.canon_mode == "body_axis_gt":
            assert self._gt_kp3d is not None, "body_axis_gt: GT not stashed"
            return self._body_axis_canon(coarse_pose,
                                         frame_src=self._gt_kp3d)
        if self.canon_mode == "frame_head":
            assert self._fh is not None, "frame_head: frame not computed"
            g, d = self._fh                              # detached
            zc = torch.zeros_like(g)
            zc[:, 2] = 1.0
            x_ax = zc - (zc * g).sum(-1, keepdim=True) * g
            xn = x_ax.norm(dim=-1, keepdim=True)
            fb = torch.zeros_like(g)
            fb[:, 0] = 1.0
            fb = fb - (fb * g).sum(-1, keepdim=True) * g
            fb = fb / fb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x_ax = torch.where(xn > 1e-3, x_ax / xn.clamp_min(1e-6), fb)
            z_ax = torch.cross(x_ax, g, dim=-1)
            R = torch.stack([x_ax, g, z_ax], dim=1)      # rows; g -> +y
            root_c = coarse_pose[:, 0].detach()
            h_root = (root_c * g).sum(-1) + d            # predicted root ht
            canon = torch.einsum(
                "bij,bkj->bki", R, coarse_pose - root_c.unsqueeze(1))
            canon = canon + torch.stack(
                [torch.zeros_like(h_root), h_root,
                 torch.zeros_like(h_root)], -1).unsqueeze(1)
            return canon
        if self._canon is not None:          # 'floor' (banned machinery era)
            R, t = self._canon[:, :3, :3], self._canon[:, :3, 3]
            return torch.einsum("bij,bkj->bki", R, coarse_pose) + t[:, None]
        return coarse_pose

    def _stash_gt(self, batch_data_samples, device):
        if self.canon_mode != "body_axis_gt":
            return
        kp = torch.cat([d.gt_instance_labels.keypoint3d
                        for d in batch_data_samples]).to(device).float()
        self._gt_kp3d = kp.reshape(-1, 16, 3)

    # Task 38: does this configuration need the temporal geometry labels?
    # (xR's H5 dataset and Quest's plain KinectEgoposeDataset have none.)
    def _needs_geom(self):
        return ((self.use_canon and self.canon_mode == "floor")
                or self.sensor_mode == "floor"
                or self.film_mode == "gravity"
                or self.lift_roll_align)

    # Task 38 P3: calibration-free canonicalization from the pose itself.
    # Frame (detached): down = root->pelvis-center; floor = farthest toe
    # projection along down; in-plane x = hip line orthogonalized to down.
    # Joint indices follow the shared xregopose-16 order (enhance_hmd_info):
    # root 0, LUpLeg 8, RUpLeg 12, LToe 11, RToe 15.
    def _body_axis_canon(self, coarse_pose, frame_src=None):
        # frame_src: pose the FRAME is derived from (Task 39 'body_axis_gt'
        # passes GT keypoint3d); default = the prediction itself (Task 38 P3).
        p = (coarse_pose if frame_src is None else frame_src).detach()
        root = p[:, 0]
        down = (p[:, 8] + p[:, 12]) * 0.5 - root
        dn = down.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(down)
        fallback[:, 2] = 1.0
        down = torch.where(dn > 1e-6, down / dn.clamp_min(1e-12), fallback)
        hip = p[:, 12] - p[:, 8]
        hip = hip - (hip * down).sum(-1, keepdim=True) * down
        hn = hip.norm(dim=-1, keepdim=True)
        fx = torch.zeros_like(hip)
        fx[:, 0] = 1.0
        fx = fx - (fx * down).sum(-1, keepdim=True) * down
        fx = fx / fx.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x_ax = torch.where(hn > 1e-6, hip / hn.clamp_min(1e-12), fx)
        up = -down
        z_ax = torch.cross(x_ax, up, dim=-1)
        R = torch.stack([x_ax, up, z_ax], dim=1)             # (B,3,3) rows
        ground = torch.maximum(((p[:, 11] - root) * down).sum(-1),
                               ((p[:, 15] - root) * down).sum(-1))
        rel = coarse_pose - root.unsqueeze(1)                # grads flow
        canon = torch.einsum("bij,bkj->bki", R, rel)
        canon = canon + torch.stack(
            [torch.zeros_like(ground), ground,
             torch.zeros_like(ground)], -1).unsqueeze(1)
        return canon

    # -- canonical transform from the per-sample device poses --------------
    def _canon_from_samples(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        c2w = torch.cat([l.temporal_cam2world for l in labels]
                        ).to(device).float()[:, 0]           # (B,4,4)
        m2w = torch.cat([l.temporal_mid2world for l in labels]
                        ).to(device).float()                 # (B,4,4)
        mask = torch.cat([l.temporal_mask for l in labels]
                         ).to(device).float()                # (B,)
        w2f = compute_relpose_to_floor(m2w) @ invert_se3(m2w)
        T = w2f @ c2w
        eye = torch.eye(4, device=device).expand_as(T)
        if self.film_mode == "gravity":
            up = torch.tensor([0.0, 1.0, 0.0], device=device)
            g = torch.einsum("bji,j->bi", c2w[:, :3, :3], up)
            g_fallback = up.expand_as(g)
            self._gvec = torch.where(mask.reshape(-1, 1) > 0, g, g_fallback)
        if self.lift_roll_align:
            if self.lift_roll_zero:
                self._roll = torch.zeros(c2w.shape[0], device=device)
            else:
                up_l = torch.tensor([0.0, 1.0, 0.0], device=device)
                g_l = torch.einsum("bji,j->bi", c2w[:, :3, :3], up_l)
                roll = torch.atan2(g_l[:, 0], g_l[:, 1])
                self._roll = torch.where(mask > 0, roll,
                                         torch.zeros_like(roll))
        if self.sensor_mode != "off":
            base_T = w2f if self.sensor_frame == "floor" else invert_se3(c2w)
            self._w2f = torch.where(mask.reshape(-1, 1, 1) > 0, base_T, eye)
            if hasattr(labels[0], "sensor_world"):
                self._sens = torch.cat(
                    [l.sensor_world for l in labels]).to(device).float()
            else:
                self._sens = torch.zeros(len(labels), 9, device=device)
        return torch.where(mask.reshape(-1, 1, 1) > 0, T, eye)

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        hm = batch_outputs[0] if isinstance(batch_outputs, tuple) \
            else batch_outputs
        self._canon = (self._canon_from_samples(batch_data_samples, hm.device)
                       if self._needs_geom() else None)
        self._stash_gt(batch_data_samples, hm.device)
        try:
            return super().decode(batch_outputs, batch_data_samples,
                                  backbone_feat)
        finally:
            self._canon = None
            self._roll = None
            self._gt_kp3d = None
            self._fh = None

    def loss(self, feats, batch_data_samples, train_cfg={}):
        self._canon = (self._canon_from_samples(
            batch_data_samples, feats[-1].device)
            if self._needs_geom() else None)
        self._stash_gt(batch_data_samples, feats[-1].device)
        self._frame_loss = None
        try:
            losses = super().loss(feats, batch_data_samples, train_cfg)
            if self.canon_mode == "frame_head" \
                    and self._frame_loss is not None:
                losses["loss_frame"] = \
                    self.frame_loss_weight * self._frame_loss
            return losses
        finally:
            self._canon = None
            self._roll = None
            self._gt_kp3d = None
            self._frame_loss = None
            self._fh = None

    def forward(self, feats):
        # parent forward with the 47x47 add_deconv output stashed for the
        # multi-scale sample (frozen stage-1 tensor; detached in refine()).
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        self._hi_feat = x
        x = self.conv_layers(x)
        return self.final_layer(x)

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
        if self.ms_mode == "off":
            spatial_feat = self.spatial_proj(sampled)
        else:
            hi_map = self._hi_feat.detach()
            if self.ms_mode == "fused_ctrl":
                hi_vec = hi_map.mean(dim=(2, 3)).unsqueeze(1).expand(
                    B, K, -1)
            else:
                hi_vec = F.grid_sample(
                    hi_map, grid, mode="bilinear", align_corners=True,
                    padding_mode="border").squeeze(2).permute(0, 2, 1)
            if self.ms_mode == "hi_only":
                spatial_feat = self.spatial_proj_ms(hi_vec)
            else:
                spatial_feat = self.spatial_proj_ms(
                    torch.cat([sampled, hi_vec], dim=-1))
        if self.film_mode != "off":
            gin = self._gvec if self.film_mode == "gravity" \
                else self.film_in.unsqueeze(0).expand(B, -1)
            gamma = 1 + self.film_gamma(gin).unsqueeze(1)
            beta = self.film_beta(gin).unsqueeze(1)
            spatial_feat = gamma * spatial_feat + beta

        pose_feat = self.pose_encoder_net(coarse_pose.reshape(B, -1))
        kin_feat = self.kin_encoder(compute_kinematic_features(coarse_pose))

        if self.use_hmd_in_refinement and hmd_info is not None:
            hmd_feat = self.hmd_encoder_stage2(hmd_info.to(torch.float32))
            hmd_exp = hmd_feat.unsqueeze(1).expand(B, K, -1)
        else:
            hmd_feat, hmd_exp = None, None

        # Task 39 Step 2: predict the frame (g_hat, d) from the image latent
        # and accumulate the soft GBH height-consistency loss.
        if self.canon_mode == "frame_head":
            raw = self.frame_head(z_latent.detach())
            g = F.normalize(raw[:, :3], dim=-1, eps=1e-6)
            d = raw[:, 3]
            if hmd_info is not None and hmd_info.shape[-1] >= 12:
                h_meas = hmd_info.to(torch.float32)[:, 9:12]     # (B,3)
                p_pts = coarse_pose.detach()[:, list(self.frame_joints)]
                h_pred = (p_pts * g.unsqueeze(1)).sum(-1) \
                    + d.unsqueeze(1) + self.frame_bias.unsqueeze(0)
                self._frame_loss = (h_pred - h_meas).abs().mean()
            self._fh = (g.detach(), d.detach())

        z_exp = z_latent.unsqueeze(1).expand(B, K, -1)
        pose_exp = pose_feat.unsqueeze(1).expand(B, K, -1)
        kin_exp = kin_feat.unsqueeze(1).expand(B, K, -1)
        coord_repr = self._apply_canon(coarse_pose) \
            if (self.attn_mode == "canon_mlp" and self.use_canon) \
            else coarse_pose
        parts = [coord_repr, spatial_feat, z_exp, pose_exp, kin_exp]
        if hmd_exp is not None:
            parts.append(hmd_exp)
        joint_input = torch.cat(parts, dim=-1)

        if self.attn_mode == "canon_mlp":
            delta = self.refinement_mlp(
                joint_input.reshape(B * K, -1)).reshape(B, K, 3)
            return coarse_pose + delta

        sens_toks = None
        if self.sensor_mode == "floor" and self._sens is not None:
            p = self._sens.reshape(B, 3, 3)              # ctrl_l, ctrl_r, hmd
            Rf, tf = self._w2f[:, :3, :3], self._w2f[:, :3, 3]
            sens_toks = torch.einsum("bij,bkj->bki", Rf, p) + tf[:, None]
        elif self.sensor_mode == "baked" and hmd_info is not None:
            h = hmd_info.to(torch.float32)
            sens_toks = torch.stack(
                [h[:, 0:3], h[:, 3:6], h[:, 9:12]], dim=1)   # (B,3,3)

        if self.attn_mode in ("replace", "replace_self", "factored"):
            canon = self._apply_canon(coarse_pose) if self.use_canon \
                else coarse_pose
            delta = self.attn_block(canon, spatial_feat, z_latent,
                                    pose_feat, kin_feat, hmd_feat,
                                    sens_toks=sens_toks)
            return coarse_pose + delta

        delta = self.refinement_mlp(
            joint_input.reshape(B * K, -1)).reshape(B, K, 3)

        # gated attention residual
        if self.attn_mode == "mlp_matched":
            extra = self.attn_block(
                joint_input.reshape(B * K, -1)).reshape(B, K, 3)
        else:
            canon = self._apply_canon(coarse_pose) if self.use_canon \
                else coarse_pose
            extra = self.attn_block(canon, spatial_feat, z_latent,
                                    pose_feat, kin_feat, hmd_feat,
                                    sens_toks=sens_toks)
        delta = delta + torch.sigmoid(self.attn_gate) * extra

        return coarse_pose + delta
