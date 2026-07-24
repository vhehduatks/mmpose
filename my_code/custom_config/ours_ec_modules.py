"""Task 26 — ego-cam-native stage 2 + 6DoF ego-cam sensor tokens, END-TO-END.

New module file (no tracked mmpose files edited; registered via
custom_imports; .reshape() only). User-specified architecture:

Stage 1: HMD fusion DROPPED (`_fuse` -> z; Task 21.4 B-none showed the
`z + hmd_linear` addition contributes ~nothing). coarse = pose_decoder(z),
ego-cam 3D.

Stage 2 (ego-cam native; alignment via a ROTATION-INVARIANT relational bias
rather than a coordinate transform):
  joint  q = joint_embed([coarse_xyz(3), canon_xyz(3), h_i(1), spatial(64)])
             + joint_pe                                       # dual-frame
  self-attn : score_ij = q_i·q_j/sqrt(d) + MLP_rel([‖p_i-p_j‖, h_i-h_j])
  cross-attn: over 2 controller tokens (HMD dropped — redundant with canon
              and Task 21.4-safe), dual-frame + 6DoF:
              sensor = [xyz_ego(3), xyz_canon(3), h(1), rot6D_canon(6)]
              score_is += MLP_relx([‖p_i-p_s‖, h_i-h_s])
  final = coarse + out(q)      # out zero-init; ego-cam delta

h_i = dot(coarse_xyz_i, g_ec) is the gravity-axis height (rotation-invariant);
g_ec = R(cam2world)^T @ [0,1,0]. rot6D_canon = first two columns of
R_canon_world @ R_world_ctrl (controller ORIENTATION — the one signal no arm
in this program has ever seen; position was credited by Task 21.2).

Controls (see configs): stage2_frame='floor' (coordinate-transform alignment,
Task-18 style; the B arm), use_bias=False (the C arm — end-to-end nocanon/E2
re-measurement), and the ablations use_sensor_ego / use_sensor_rot.

Init: from the headline lifter checkpoint (backbone + encoder + pose_decoder);
stage-2 is new (out zero-init). No frozen-stage identity is possible (fusion
removed), so this is a full end-to-end retrain like Task 24.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, soft_argmax_2d)
from mmpose.registry import DATASETS, KEYPOINT_CODECS, MODELS

from my_code.custom_config.ours_t_modules import (  # noqa: F401
    KinectEgoposeTemporalDataset, OursTemporalCodec,
    compute_relpose_to_floor, invert_se3)


# ---------------------------------------------------------------------------
# Data: sensors_v3 = v2 positions (T,9) + controller world rotations (T,2,3,3)
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgoposeSensorV3Dataset(KinectEgoposeTemporalDataset):
    """Temporal dataset + per-sample controller world positions AND rotations
    from sensors_v3.npz (ours/frame_adapt/gen_sensors_v3.py). Adds
    `sensor_world` (1,9) and `sensor_rot_world` (1,18)=flattened (2,3,3)
    [R_world_ctrl_left, R_world_ctrl_right] in the y-up FRAME world. Missing
    session/frame -> zeros / identity."""

    def __init__(self, *, sensors_file: str = "sensors_v3.npz", **kwargs):
        self.sensors_file = sensors_file
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        cache = {}
        n_missing = 0
        eye18 = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (2, 1)
                        ).reshape(1, 18)
        for d in data_list:
            img = d["img_path"]
            fid = int(pat.search(img).group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            key = (participant, session)
            if key not in cache:
                p = (Path(self.frame_export_root) / participant / "actions"
                     / session / self.sensors_file)
                if p.is_file():
                    z = np.load(p)
                    cache[key] = (z["sensors"].astype(np.float32),
                                  z["ctrl_rot"].astype(np.float32))
                else:
                    cache[key] = None
            arr = cache[key]
            if arr is not None and fid < len(arr[0]):
                d["sensor_world"] = arr[0][fid].reshape(1, 9)
                d["sensor_rot_world"] = arr[1][fid].reshape(1, 18)
            else:
                d["sensor_world"] = np.zeros((1, 9), dtype=np.float32)
                d["sensor_rot_world"] = eye18.copy()
                n_missing += 1
        print(f"[KinectEgoposeSensorV3Dataset] sensors_file="
              f"{self.sensors_file}, {n_missing} samples without sensors")
        return data_list


@KEYPOINT_CODECS.register_module()
class OursSensorV3Codec(OursTemporalCodec):
    label_mapping_table = dict(
        OursTemporalCodec.label_mapping_table,
        sensor_world="sensor_world",
        sensor_rot_world="sensor_rot_world",
    )


# ---------------------------------------------------------------------------
# Multi-head attention with an additive per-head relational bias
# ---------------------------------------------------------------------------

class BiasedMHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, xq, xkv, bias=None):
        B, Lq, _ = xq.shape
        Lk = xkv.shape[1]
        q = self.q(xq).reshape(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).reshape(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).reshape(B, Lk, self.h, self.dh).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)
        if bias is not None:                      # bias (B, H, Lq, Lk)
            s = s + bias
        a = torch.softmax(s, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, Lq, self.h * self.dh)
        return self.o(o)


def _rel_mlp(heads):
    """[dist, dheight] (..,2) -> per-head bias (..,heads); last layer zero-init
    so the block starts as plain attention."""
    m = nn.Sequential(nn.Linear(2, 32), nn.ReLU(inplace=True),
                      nn.Linear(32, heads))
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class OursEgoCamHead(CustomEgoposeCascadedRefinementHead_enhanced):

    def __init__(self, *args,
                 stage2_frame: str = "egocam",   # 'egocam' | 'floor'
                 use_bias: bool = True,
                 use_sensor_ego: bool = True,
                 use_sensor_rot: bool = True,
                 d: int = 64, heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        assert stage2_frame in ("egocam", "floor"), stage2_frame
        self.stage2_frame = stage2_frame
        self.use_bias = bool(use_bias)
        self.use_sensor_ego = bool(use_sensor_ego)
        self.use_sensor_rot = bool(use_sensor_rot)
        self.d = d

        # stage-1 fusion removal: z_plus_hmd = z  (Task 21.4 B-none)
        self._fuse = lambda z, hmd_emb: z

        # spatial feature: sample backbone_feat (2048ch) at soft-argmax joints
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.in_channels, d), nn.ReLU(inplace=True))

        # joint token embedding (dual-frame in egocam, canon-only in floor)
        j_in = (3 + 3 + 1 + d) if stage2_frame == "egocam" else (3 + 1 + d)
        self.joint_embed = nn.Sequential(nn.Linear(j_in, d),
                                         nn.ReLU(inplace=True))
        self.joint_pe = nn.Parameter(torch.zeros(16, d))
        nn.init.normal_(self.joint_pe, std=0.02)

        # sensor token embedding (2 controllers; HMD dropped)
        s_in = (3 if use_sensor_ego else 0) + 3 + 1 + (6 if use_sensor_rot else 0)
        self.sens_embed = nn.Sequential(nn.Linear(s_in, d),
                                        nn.ReLU(inplace=True))
        self.sens_type = nn.Parameter(torch.zeros(2, d))
        nn.init.normal_(self.sens_type, std=0.02)

        # invariant relational bias
        if use_bias:
            self.mlp_rel = _rel_mlp(heads)
            self.mlp_relx = _rel_mlp(heads)

        # attention block
        self.self_attn = BiasedMHA(d, heads)
        self.cross_attn = BiasedMHA(d, heads)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(inplace=True),
                                 nn.Linear(4 * d, d))
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight)      # delta starts at 0 -> final=coarse
        nn.init.zeros_(self.out.bias)

    # ---- per-batch geometry (stashed for refine, like OursAttnCascadedHead) --
    def _prep(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        c2w = torch.cat([l.temporal_cam2world for l in labels]
                        ).to(device).float()[:, 0]           # (B,4,4)
        m2w = torch.cat([l.temporal_mid2world for l in labels]
                        ).to(device).float()                 # (B,4,4)
        mask = torch.cat([l.temporal_mask for l in labels]
                         ).to(device).float().reshape(-1)    # (B,)
        B = c2w.shape[0]
        eye4 = torch.eye(4, device=device).expand(B, 4, 4)
        m = mask.reshape(-1, 1, 1) > 0

        w2f = compute_relpose_to_floor(m2w) @ invert_se3(m2w)   # world->floor
        w2c = invert_se3(c2w)                                   # world->cam
        canon_ej = w2f @ c2w                                    # ego->floor
        self._w2f = torch.where(m, w2f, eye4)
        self._w2c = torch.where(m, w2c, eye4)
        self._canon = torch.where(m, canon_ej, eye4)

        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        g = torch.einsum("bji,j->bi", c2w[:, :3, :3], up)      # R_c2w^T @ up
        self._g = torch.where(mask.reshape(-1, 1) > 0, g, up.expand(B, 3))

        if hasattr(labels[0], "sensor_world"):
            sens = torch.cat([l.sensor_world for l in labels]
                             ).to(device).float().reshape(B, 3, 3)[:, :2]  # L,R
        else:
            sens = torch.zeros(B, 2, 3, device=device)
        self._sens_world = sens
        if hasattr(labels[0], "sensor_rot_world"):
            srot = torch.cat([l.sensor_rot_world for l in labels]
                             ).to(device).float().reshape(B, 2, 3, 3)
        else:
            srot = torch.eye(3, device=device).reshape(1, 1, 3, 3).expand(
                B, 2, 3, 3)
        self._sens_rot_world = srot

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        hm = batch_outputs[0] if isinstance(batch_outputs, tuple) \
            else batch_outputs
        self._prep(batch_data_samples, hm.device)
        try:
            return super().decode(batch_outputs, batch_data_samples,
                                  backbone_feat)
        finally:
            self._sens_world = None

    def loss(self, feats, batch_data_samples, train_cfg={}):
        self._prep(batch_data_samples, feats[-1].device)
        try:
            return super().loss(feats, batch_data_samples, train_cfg)
        finally:
            self._sens_world = None

    def _sensor_tokens(self, coarse_pose):
        B = coarse_pose.shape[0]
        Rw2c, tw2c = self._w2c[:, :3, :3], self._w2c[:, :3, 3]
        Rw2f, tw2f = self._w2f[:, :3, :3], self._w2f[:, :3, 3]
        p_ego = torch.einsum("bij,bsj->bsi", Rw2c, self._sens_world) \
            + tw2c[:, None]                                    # (B,2,3)
        p_canon = torch.einsum("bij,bsj->bsi", Rw2f, self._sens_world) \
            + tw2f[:, None]
        h_s = torch.einsum("bsj,bj->bs", p_ego, self._g)       # (B,2)
        parts = []
        if self.use_sensor_ego:
            parts.append(p_ego)
        parts.append(p_canon)
        parts.append(h_s.unsqueeze(-1))
        if self.use_sensor_rot:
            R_canon_ctrl = torch.einsum(
                "bij,bsjk->bsik", Rw2f, self._sens_rot_world)  # (B,2,3,3)
            rot6d = R_canon_ctrl[..., :2].reshape(B, 2, 6)     # first 2 cols
            parts.append(rot6d)
        tok = torch.cat(parts, dim=-1)
        kv = self.sens_embed(tok) + self.sens_type.unsqueeze(0)
        return kv, p_ego, h_s

    def refine(self, coarse_pose, heatmap, backbone_feat, z_latent,
               hmd_info=None):
        B, K = coarse_pose.shape[0], coarse_pose.shape[1]

        hm = heatmap.detach()
        coords, _ = soft_argmax_2d(hm, temperature=0.1)
        grid = (coords * 2 - 1).unsqueeze(1)
        sampled = F.grid_sample(backbone_feat, grid, mode="bilinear",
                                align_corners=True, padding_mode="border")
        spatial = self.spatial_proj(sampled.squeeze(2).permute(0, 2, 1))

        g = self._g                                            # (B,3)
        h_i = torch.einsum("bkj,bj->bk", coarse_pose, g)       # (B,16)
        R, t = self._canon[:, :3, :3], self._canon[:, :3, 3]
        canon_xyz = torch.einsum("bij,bkj->bki", R, coarse_pose) + t[:, None]

        if self.stage2_frame == "egocam":
            feats = [coarse_pose, canon_xyz, h_i.unsqueeze(-1), spatial]
        else:
            feats = [canon_xyz, h_i.unsqueeze(-1), spatial]
        q = self.joint_embed(torch.cat(feats, dim=-1)) + self.joint_pe

        kv, p_ego, h_s = self._sensor_tokens(coarse_pose)

        bias_self = bias_cross = None
        if self.use_bias:
            d_ij = torch.cdist(coarse_pose, coarse_pose)       # (B,16,16)
            dh_ij = h_i[:, :, None] - h_i[:, None, :]
            bias_self = self.mlp_rel(
                torch.stack([d_ij, dh_ij], dim=-1)).permute(0, 3, 1, 2)
            d_is = torch.cdist(coarse_pose, p_ego)             # (B,16,2)
            dh_is = h_i[:, :, None] - h_s[:, None, :]
            bias_cross = self.mlp_relx(
                torch.stack([d_is, dh_is], dim=-1)).permute(0, 3, 1, 2)

        q = self.ln1(q + self.self_attn(q, q, bias_self))
        q = self.ln2(q + self.cross_attn(q, kv, bias_cross))
        q = self.ln3(q + self.ffn(q))
        delta = self.out(q)                                    # (B,16,3)

        if self.stage2_frame == "floor":                       # floor -> ego
            delta = torch.einsum("bji,bkj->bki", R, delta)     # R^T @ delta
        return coarse_pose + delta
