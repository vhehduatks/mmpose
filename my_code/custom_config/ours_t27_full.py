"""Task 27-FULL — unified spatio-temporal model over the 26a feature cache.

Frozen: 26a backbone + stage-1 (consumed via the per-joint sampled cache from
ours_t27_cache.py). Trainable: 26a's ego-cam stage-2 (init from the 26a ckpt)
+ a MixSTE-style joint-axis temporal transformer consuming the stage-2's q
features (16x64) per frame:

  tokens (B,T=20,J=16,d): tok = proj(q) [arm F] + coord_embed(p_refined) + PEs
  [STB: attn over J] <-> [TTB: attn over T] x N pairs; relative-cam-pose
  (T_t -> T_last, rot6D+trans) conditioning added before each TTB.
  output: zero-init head on last-step tokens -> final = refined_last + dp.

Arms: F (q, stage-2 jointly trained) | Fcoord (coords only, joint) |
Ffrozen (q, stage-2 frozen). Loss = ||final - gt_last|| + 0.5 * anchor
(mean_t ||refined_t - gt_t||, keeps out(q) meaningful).

Run (mmpose env):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python \
        my_code/custom_config/ours_t27_full.py --arm F
Gates:  --gate identity   (zero-init head => F == cached 26a refined)
        --gate mem        (fwd+bwd peak memory at batch size)
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import MODELS

from mmpose.models.heads.heatmap_heads.\
    custom_egopose_cascaded_refinement_head_enhanced import (
        compute_kinematic_features)

from my_code.custom_config.ours_ec_modules import OursEgoCamHead  # noqa: F401
from my_code.custom_config.ours_t_modules import (compute_relpose_to_floor,
                                                  invert_se3)

# Task 28: the deployed cached_noise recipe (framevision cached_noise.yaml,
# fitted from fold-vs-pilot residuals): AR(1) rho=0.961, per-axis std =
# sigma_3DRMS/sqrt(3), wrist-heavy per-joint scales (mean ~1).
AR1_RHO = 0.961
JOINT_SCALE = [0.62, 0.62, 0.62, 1.09, 1.66, 0.66, 1.11, 1.44,
               0.68, 0.92, 1.20, 1.31, 0.69, 0.91, 1.13, 1.34]

CFG = "my_code/custom_config/HMD_kinect_v5_t26_ecA_config.py"
CKPT = ("work_dirs/t26_ecA/"
        "best_xregopose_Full Body_All_mpjpe_epoch_6.pth")
CACHE = "/mnt/dataset_vol/t27_feat_cache"
T = 20

STAGE2_MODULES = ("spatial_proj", "joint_embed", "sens_embed", "mlp_rel",
                  "mlp_relx", "self_attn", "cross_attn", "ln1", "ln2", "ln3",
                  "ffn", "out")
STAGE2_PARAMS = ("joint_pe", "sens_type")


# ---------------------------------------------------------------------------
# Data: windows of T consecutive cache entries per session
# ---------------------------------------------------------------------------

class WindowData:
    def __init__(self, split, root=CACHE):
        self.sessions = []
        for f in sorted(glob.glob(os.path.join(root, split, "*.npz"))):
            z = np.load(f)
            self.sessions.append({k: z[k] for k in z.files})
        self.index = []
        for si, s in enumerate(self.sessions):
            n = len(s["fid"])
            for end in range(T - 1, n):
                self.index.append((si, end))

    def __len__(self):
        return len(self.index)

    def batch(self, ids, device):
        rows = [self.index[i] for i in ids]
        out = {}
        for key, dt in (("sampled", torch.float32), ("z", torch.float32),
                        ("coarse", torch.float32), ("refined", torch.float32),
                        ("gt", torch.float32), ("c2w", torch.float32),
                        ("m2w", torch.float32), ("mask", torch.float32),
                        ("sens", torch.float32), ("srot", torch.float32)):
            arr = np.stack([self.sessions[si][key][end - T + 1:end + 1]
                            for si, end in rows])
            out[key] = torch.from_numpy(arr).to(device).to(dt)
        return out


# ---------------------------------------------------------------------------
# MixSTE-style temporal module
# ---------------------------------------------------------------------------

class AttnBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):                       # (N, L, d)
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))


class TemporalMixSTE(nn.Module):
    def __init__(self, use_q=True, q_dim=64, d=128, heads=8, pairs=4,
                 dropout=0.1, dual_frame=False, use_sensors=False,
                 globals_mode=None, use_rel9=True):
        super().__init__()
        self.use_q = use_q
        self.dual_frame = dual_frame
        self.use_sensors = use_sensors
        self.globals_mode = globals_mode
        self.use_rel9 = use_rel9
        if globals_mode:
            # Task 29: pesens global K/V restored (minus mod_hmd) as extra
            # tokens in the SAME weight-tied per-STB cross-attn; zero-init
            # out_proj keeps the identity gate. NEW encoders (the head's
            # pose_encoder_net/kin_encoder are headline-era, stale).
            assert use_sensors, "globals ride the sensor cross-attn K/V"
            self.n_glob = 1 if globals_mode == "z" else 3
            if globals_mode == "const":
                self.glob_const = nn.Parameter(torch.zeros(3, d))
                nn.init.normal_(self.glob_const, std=0.02)
            else:
                self.glob_z = nn.Linear(64, d)
                if globals_mode == "all":
                    self.glob_pose = nn.Linear(48, d)
                    self.glob_kin = nn.Linear(60, d)
        if use_q:
            self.q_proj = nn.Linear(q_dim, d)
        # Task 28 lever 0: dual-frame token [p_ego, p_common, h] (7-d)
        self.coord_embed = nn.Linear(7 if dual_frame else 3, d)
        if use_sensors:
            # Task 28 lever 2: per-step controller K/V [s_ego, s_common, h],
            # ONE weight-tied cross-attn applied after each STB; zero-init
            # out_proj => no-op at init (identity gate preserved).
            self.sens_embed = nn.Linear(7, d)
            n_kv = 2 + (self.n_glob if globals_mode else 0)
            self.sens_type = nn.Parameter(torch.zeros(n_kv, d))
            nn.init.normal_(self.sens_type, std=0.02)
            self.sens_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                   batch_first=True)
            nn.init.zeros_(self.sens_attn.out_proj.weight)
            nn.init.zeros_(self.sens_attn.out_proj.bias)
        self.joint_pe = nn.Parameter(torch.zeros(16, d))
        self.time_pe = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.joint_pe, std=0.02)
        nn.init.normal_(self.time_pe, std=0.02)
        if use_rel9:
            self.rel_embed = nn.Sequential(nn.Linear(9, d), nn.GELU(),
                                           nn.Linear(d, d))
        self.stb = nn.ModuleList(AttnBlock(d, heads, dropout)
                                 for _ in range(pairs))
        self.ttb = nn.ModuleList(AttnBlock(d, heads, dropout)
                                 for _ in range(pairs))
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, p_tok, rel9, sens_tok=None, glob=None):
        # q (B,T,16,qd) or None; p_tok (B,T,16,3|7); rel9 (B,T,9);
        # sens_tok (B,T,2,7) or None; glob = (z(B,T,64), pflat(B,T,48),
        # kin(B,T,60)) or None
        B = p_tok.shape[0]
        x = self.coord_embed(p_tok)
        if self.use_q:
            x = x + self.q_proj(q)
        x = x + self.joint_pe + self.time_pe[None, :, None]
        # Task 30 A6: rel9 conditioning ablatable
        cond = (self.rel_embed(rel9)[:, :, None]             # (B,T,1,d)
                if self.use_rel9 else 0.0)
        kv = None
        if self.use_sensors and sens_tok is not None:
            kv = self.sens_embed(sens_tok)                   # (B,T,2,d)
            if self.globals_mode == "const":
                kv = torch.cat([kv, self.glob_const.expand(
                    B, T, 3, -1)], dim=2)
            elif self.globals_mode:
                g = [self.glob_z(glob[0])]
                if self.globals_mode == "all":
                    g += [self.glob_pose(glob[1]), self.glob_kin(glob[2])]
                kv = torch.cat([kv, torch.stack(g, dim=2)], dim=2)
            kv = (kv + self.sens_type).reshape(B * T, kv.shape[2], -1)
        for stb, ttb in zip(self.stb, self.ttb):
            xf = stb(x.reshape(B * T, 16, -1))
            if kv is not None:
                a, _ = self.sens_attn(xf, kv, kv)
                xf = xf + a                    # zero-init out_proj: 0 at init
            x = xf.reshape(B, T, 16, -1) + cond
            x = ttb(x.permute(0, 2, 1, 3).reshape(B * 16, T, -1)
                    ).reshape(B, 16, T, -1).permute(0, 2, 1, 3)
        return self.out(x)                             # (B,T,16,3) all-step


class FlattenTemporal(nn.Module):
    """Task 30 A7 — param-matched flatten control (PoseFormer-style temporal
    stage): the 16 joints collapse to ONE token per step, so ONLY the joint
    axis differs from TemporalMixSTE. 8 temporal AttnBlocks at the same d as
    the 4 STB+4 TTB pairs (same block count => params match within ~5%);
    weight-tied sensor cross-attn (single query token) after every 2nd block
    = 4 applications, matching A0; rel9 added before each block; zero-init
    out => identity gate holds."""

    def __init__(self, use_q=True, q_dim=64, d=128, heads=8, blocks=8,
                 dropout=0.1, dual_frame=False, use_sensors=False,
                 use_rel9=True):
        super().__init__()
        self.use_q = use_q
        self.use_sensors = use_sensors
        self.use_rel9 = use_rel9
        in_dim = 16 * (7 if dual_frame else 3) + (q_dim if use_q else 0)
        self.step_embed = nn.Linear(in_dim, d)
        self.time_pe = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.time_pe, std=0.02)
        if use_rel9:
            self.rel_embed = nn.Sequential(nn.Linear(9, d), nn.GELU(),
                                           nn.Linear(d, d))
        if use_sensors:
            self.sens_embed = nn.Linear(7, d)
            self.sens_type = nn.Parameter(torch.zeros(2, d))
            nn.init.normal_(self.sens_type, std=0.02)
            self.sens_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                   batch_first=True)
            nn.init.zeros_(self.sens_attn.out_proj.weight)
            nn.init.zeros_(self.sens_attn.out_proj.bias)
        self.blocks = nn.ModuleList(AttnBlock(d, heads, dropout)
                                    for _ in range(blocks))
        self.out = nn.Linear(d, 48)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, p_tok, rel9, sens_tok=None, glob=None):
        B, Tn = p_tok.shape[:2]
        feats = p_tok.reshape(B, Tn, -1)                 # (B,T,16*pd)
        if self.use_q:
            feats = torch.cat([feats, q.mean(dim=2)], dim=-1)  # + q-bar
        x = self.step_embed(feats) + self.time_pe[None]
        cond = self.rel_embed(rel9) if self.use_rel9 else 0.0
        kv = None
        if self.use_sensors and sens_tok is not None:
            kv = (self.sens_embed(sens_tok) + self.sens_type
                  ).reshape(B * Tn, 2, -1)
        for i, blk in enumerate(self.blocks):
            x = blk(x + cond)
            if kv is not None and i % 2 == 1:
                a, _ = self.sens_attn(x.reshape(B * Tn, 1, -1), kv, kv)
                x = x + a.reshape(B, Tn, -1)
        return self.out(x).reshape(B, Tn, 16, 3)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class T27Full(nn.Module):
    def __init__(self, head, arm, noise_sigma=0.0, noise_mode="white_legacy",
                 dual_frame=False, use_sensors=False, globals_mode=None,
                 use_rel9=True, flatten=False):
        super().__init__()
        self.head = head
        self.arm = arm
        self.noise_sigma = noise_sigma          # mm, train-time input noise
        self.noise_mode = noise_mode  # white_legacy | white | ar1
        self.dual_frame = dual_frame
        self.use_sensors = use_sensors
        self.globals_mode = globals_mode
        if noise_mode == "ar1":
            self.register_buffer("jscale", torch.tensor(
                JOINT_SCALE).reshape(1, 1, 16, 1))
        self.frozen = arm in ("Ffrozen", "Ffrozencoord")
        cls = FlattenTemporal if flatten else TemporalMixSTE
        kw = {} if flatten else {"globals_mode": globals_mode}
        self.temporal = cls(
            use_q=arm not in ("Fcoord", "Ffrozencoord"),
            dual_frame=dual_frame, use_sensors=use_sensors,
            use_rel9=use_rel9, **kw)
        if head is None:                 # Task 31: coords straight from cache
            assert arm == "Ffrozencoord", "no head => coords-only arm"
            return
        for p in head.parameters():
            p.requires_grad_(False)
        if not self.frozen:
            for name in STAGE2_MODULES:
                m = getattr(head, name, None)
                if isinstance(m, nn.Module):
                    m.requires_grad_(True)
            for name in STAGE2_PARAMS:
                p = getattr(head, name, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad_(True)

    def stage2(self, b):
        B = b["coarse"].shape[0]
        flat = lambda x: x.reshape(B * T, *x.shape[2:])  # noqa: E731
        self.head._prep_from_geom(flat(b["c2w"]), flat(b["m2w"]),
                                  flat(b["mask"]), flat(b["sens"]),
                                  flat(b["srot"]))
        refined, q = self.head._refine_core(flat(b["coarse"]),
                                            flat(b["sampled"]), flat(b["z"]))
        return (refined.reshape(B, T, 16, 3), q.reshape(B, T, 16, -1))

    def _make_noise(self, refined):
        """Train-time input noise, deployed cached_noise conventions:
        sigma = 3D-RMS mm => per-axis std sigma/sqrt(3); 'ar1' adds rho=0.961
        temporal correlation + wrist-heavy per-joint scales. 'white_legacy'
        keeps the 27-full probe convention (per-axis sigma mm) for
        reproducibility of the recorded 55.44."""
        if self.noise_mode == "white_legacy":
            return torch.randn_like(refined) * (self.noise_sigma / 1000.0)
        s = self.noise_sigma / 1000.0 / math.sqrt(3.0)
        eps = torch.randn_like(refined) * s              # (B,T,16,3)
        if self.noise_mode == "white":
            return eps
        n = torch.empty_like(eps)
        n[:, 0] = eps[:, 0]
        c = math.sqrt(1.0 - AR1_RHO ** 2)
        for t in range(1, refined.shape[1]):
            n[:, t] = AR1_RHO * n[:, t - 1] + c * eps[:, t]
        return n * self.jscale

    def forward(self, b):
        if self.head is None:            # Task 31: cached refined, no q
            refined, q = b["refined"], None
        elif self.frozen:
            with torch.no_grad():
                refined, q = self.stage2(b)
        else:
            refined, q = self.stage2(b)
        anchor = refined
        if self.training and self.noise_sigma > 0:
            # noisy tokens AND noisy residual base, clean targets; test-time
            # inputs stay clean (noise-to-test, deployed STF2 discipline).
            refined = refined + self._make_noise(refined)
        B = refined.shape[0]
        c2w = b["c2w"]
        eye = torch.eye(4, device=c2w.device)
        m = b["mask"].reshape(B, T, 1, 1) > 0
        rel = torch.matmul(invert_se3(
            c2w[:, -1].reshape(B, 4, 4)).unsqueeze(1), c2w)    # (B,T,4,4)
        rel = torch.where(m & m[:, -1:], rel, eye.expand_as(rel))
        rel9 = torch.cat([rel[..., :3, :2].reshape(B, T, 6),
                          rel[..., :3, 3]], dim=-1)

        p_tok, sens_tok = refined, None
        if self.dual_frame or self.use_sensors:
            # T_t = w2f_last @ c2w_t maps step-t ego-cam -> last-step floor
            m2w_l = b["m2w"][:, -1].reshape(B, 4, 4)
            w2f_l = compute_relpose_to_floor(m2w_l) @ invert_se3(m2w_l)
            Tt = torch.matmul(w2f_l.unsqueeze(1), c2w)         # (B,T,4,4)
            Tt = torch.where(m & m[:, -1:], Tt, eye.expand_as(Tt))
            up = torch.tensor([0.0, 1.0, 0.0], device=c2w.device)
            g = torch.einsum("btji,j->bti", c2w[..., :3, :3], up)
            g = torch.where(m.reshape(B, T, 1) > 0, g,
                            up.expand(B, T, 3))                # (B,T,3)
        if self.dual_frame:
            p_com = torch.einsum("btij,btkj->btki", Tt[..., :3, :3],
                                 refined) + Tt[..., None, :3, 3]
            h = torch.einsum("btkj,btj->btk", refined, g)
            p_tok = torch.cat([refined, p_com, h.unsqueeze(-1)], dim=-1)
        if self.use_sensors:
            sw = b["sens"]                                     # (B,T,2,3) wrld
            w2c = invert_se3(c2w.reshape(B * T, 4, 4)).reshape(B, T, 4, 4)
            s_ego = torch.einsum("btij,btsj->btsi", w2c[..., :3, :3],
                                 sw) + w2c[..., None, :3, 3]
            s_com = torch.einsum("btij,btsj->btsi",
                                 w2f_l[:, None, :3, :3].expand(B, T, 3, 3),
                                 sw) + w2f_l[:, None, None, :3, 3]
            h_s = torch.einsum("btsj,btj->bts", s_ego, g)
            sens_tok = torch.cat([s_ego, s_com, h_s.unsqueeze(-1)], dim=-1)

        glob = None
        if self.globals_mode and self.globals_mode != "const":
            # globals see the NOISED stream (same as the joint tokens; the
            # noise discipline must not leak clean coords through g_pose/kin)
            glob = (b["z"], refined.reshape(B, T, 48),
                    compute_kinematic_features(
                        refined.reshape(B * T, 16, 3)).reshape(B, T, 60))
        delta_all = self.temporal(q, p_tok, rel9, sens_tok, glob)  # (B,T,16,3)
        final_all = refined + delta_all
        # anchor (2nd return) stays the CLEAN refined for the aux loss
        return final_all[:, -1], anchor, final_all


def run_val(model, data, device, bs):
    model.eval()
    errs, base = [], []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            ids = range(i, min(i + bs, len(data)))
            b = data.batch(list(ids), device)
            final, _, _ = model(b)
            gt = b["gt"][:, -1]
            errs.append((final - gt).norm(dim=-1).mean(-1).cpu() * 1000)
            base.append((b["refined"][:, -1] - gt
                         ).norm(dim=-1).mean(-1).cpu() * 1000)
    return (torch.cat(errs).mean().item(), torch.cat(base).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F", "Fcoord", "Ffrozen",
                                     "Ffrozencoord"], required=True)
    ap.add_argument("--gate", choices=["identity", "mem"], default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr-temporal", type=float, default=3e-4)
    ap.add_argument("--lr-stage2", type=float, default=1e-4)
    ap.add_argument("--noise-sigma", type=float, default=0.0)
    ap.add_argument("--noise-mode", default="white_legacy",
                    choices=["white_legacy", "white", "ar1"])
    ap.add_argument("--dual-frame", action="store_true")
    ap.add_argument("--sensors", action="store_true")
    ap.add_argument("--vel-loss", action="store_true")
    ap.add_argument("--globals", dest="globals_mode", default=None,
                    choices=["all", "z", "const"])
    ap.add_argument("--no-rel9", action="store_true")      # Task 30 A6
    ap.add_argument("--flatten", action="store_true")      # Task 30 A7
    ap.add_argument("--seed", type=int, default=None)      # Task 30: init+shuffle
    ap.add_argument("--cache", default=CACHE)              # Task 30 B: alt cache
    ap.add_argument("--tag-prefix", default="")            # Task 30 B: run tag
    ap.add_argument("--cfg", default=CFG)                  # Task 30 B: alt head
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--coords-from-cache", action="store_true")  # Task 31
    ap.add_argument("--eval-ckpt", default=None)   # Task 31: eval-only mode
    ap.add_argument("--eval-split", default="Val")
    args = ap.parse_args()
    device = "cuda"
    if args.seed is not None:
        torch.manual_seed(args.seed)   # varies module init (pre-build)

    init_default_scope("mmpose")
    if args.coords_from_cache:           # Task 31: no stage-2 head at all
        head = None
    else:
        cfg = Config.fromfile(args.cfg)
        head = MODELS.build(cfg.model["head"])
        sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        head.load_state_dict({k[5:]: v for k, v in sd.items()
                              if k.startswith("head.")}, strict=True)
        head = head.to(device)
    model = T27Full(head, args.arm,
                    noise_sigma=args.noise_sigma, noise_mode=args.noise_mode,
                    dual_frame=args.dual_frame, use_sensors=args.sensors,
                    globals_mode=args.globals_mode,
                    use_rel9=not args.no_rel9,
                    flatten=args.flatten).to(device)
    n_tmp = sum(p.numel() for p in model.temporal.parameters())
    print(f"temporal params: {n_tmp/1e6:.3f}M "
          f"({type(model.temporal).__name__})", flush=True)

    if args.eval_ckpt:                 # Task 31: single predefined eval pass
        data = WindowData(args.eval_split, args.cache)
        print(f"{args.eval_split} windows: {len(data)}", flush=True)
        ck = torch.load(args.eval_ckpt, map_location="cpu")
        model.load_state_dict(ck["model"])
        mp, bp = run_val(model, data, device, args.bs)
        print(f"EVAL[{args.eval_split}] {args.eval_ckpt} "
              f"(train-best ep{ck.get('epoch')}, {ck.get('val_mpjpe'):.2f}): "
              f"MPJPE {mp:.2f} mm, base {bp:.2f} mm, "
              f"extracted {mp-bp:+.2f}", flush=True)
        return

    val = WindowData("Val", args.cache)
    print(f"val windows: {len(val)}", flush=True)

    if args.gate == "identity":
        b = val.batch(list(range(64)), device)
        model.eval()
        with torch.no_grad():
            final, refined, _ = model(b)
        d1 = float((final - refined[:, -1]).abs().max())
        d2 = float((refined[:, -1] - b["refined"][:, -1]).abs().max()) * 1000
        print(f"IDENTITY: |final-refined_last| = {d1:.3e} (want 0); "
              f"|recomputed-cached refined| = {d2:.4f} mm (fp16 tol)")
        mp, bp = run_val(model, val, device, args.bs)
        print(f"IDENTITY val MPJPE {mp:.2f} vs cached-26a {bp:.2f} mm")
        return
    if args.gate == "mem":
        train = val                                    # shape-equivalent
        b = train.batch(list(range(args.bs)), device)
        final, refined, fall = model(b)
        loss = (final - b["gt"][:, -1]).norm(dim=-1).mean() \
            + 0.5 * (refined - b["gt"]).norm(dim=-1).mean() \
            + 0.5 * ((fall[:, 1:] - fall[:, :-1])
                     - (b["gt"][:, 1:] - b["gt"][:, :-1])
                     ).norm(dim=-1).mean()
        loss.backward()
        print(f"MEM: peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"at bs={args.bs}")
        return

    train = WindowData("Train", args.cache)
    print(f"train windows: {len(train)}", flush=True)
    groups = [{"params": [p for p in model.temporal.parameters()],
               "lr": args.lr_temporal}]
    s2 = ([] if model.head is None else
          [p for p in model.head.parameters() if p.requires_grad])
    if s2:
        groups.append({"params": s2, "lr": args.lr_stage2})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [6, 8], 0.5)

    tag = (args.tag_prefix + f"{args.arm}"
           + (f"_{args.noise_mode}{args.noise_sigma:g}"
              if args.noise_sigma else "")
           + ("_df" if args.dual_frame else "")
           + ("_sens" if args.sensors else "")
           + ("_vel" if args.vel_loss else "")
           + (f"_glob{args.globals_mode}" if args.globals_mode else "")
           + ("_norel9" if args.no_rel9 else "")
           + ("_flat" if args.flatten else "")
           + (f"_s{args.seed}" if args.seed is not None else ""))
    out_dir = f"work_dirs/t27_full_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    best = math.inf
    rng = np.random.default_rng(42 if args.seed is None else args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(train))
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            b = train.batch(perm[i:i + args.bs].tolist(), device)
            final, refined, fall = model(b)
            loss = (final - b["gt"][:, -1]).norm(dim=-1).mean() \
                + 0.5 * (refined - b["gt"]).norm(dim=-1).mean()
            if args.vel_loss:
                loss = loss + 0.5 * (
                    (fall[:, 1:] - fall[:, :-1])
                    - (b["gt"][:, 1:] - b["gt"][:, :-1])
                ).norm(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            if nb % 400 == 0:
                print(f"ep{ep} it{nb} loss {tot/nb:.4f}", flush=True)
        sched.step()
        mp, bp = run_val(model, val, device, args.bs)
        print(f"== ep{ep} val MPJPE {mp:.2f} mm  (26a baseline {bp:.2f}, "
              f"extracted {mp-bp:+.2f})", flush=True)
        if mp < best:
            best = mp
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_mpjpe": mp}, f"{out_dir}/best.pth")
    print(f"BEST {tag}: {best:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
