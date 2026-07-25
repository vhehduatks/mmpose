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

from my_code.custom_config.ours_ec_modules import OursEgoCamHead  # noqa: F401
from my_code.custom_config.ours_t_modules import invert_se3

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
    def __init__(self, split):
        self.sessions = []
        for f in sorted(glob.glob(os.path.join(CACHE, split, "*.npz"))):
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
                 dropout=0.1):
        super().__init__()
        self.use_q = use_q
        if use_q:
            self.q_proj = nn.Linear(q_dim, d)
        self.coord_embed = nn.Linear(3, d)
        self.joint_pe = nn.Parameter(torch.zeros(16, d))
        self.time_pe = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.joint_pe, std=0.02)
        nn.init.normal_(self.time_pe, std=0.02)
        self.rel_embed = nn.Sequential(nn.Linear(9, d), nn.GELU(),
                                       nn.Linear(d, d))
        self.stb = nn.ModuleList(AttnBlock(d, heads, dropout)
                                 for _ in range(pairs))
        self.ttb = nn.ModuleList(AttnBlock(d, heads, dropout)
                                 for _ in range(pairs))
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, p, rel9):
        # q (B,T,16,qd) or None, p (B,T,16,3), rel9 (B,T,9)
        B = p.shape[0]
        x = self.coord_embed(p)
        if self.use_q:
            x = x + self.q_proj(q)
        x = x + self.joint_pe + self.time_pe[None, :, None]
        cond = self.rel_embed(rel9)[:, :, None]              # (B,T,1,d)
        for stb, ttb in zip(self.stb, self.ttb):
            x = stb(x.reshape(B * T, 16, -1)).reshape(B, T, 16, -1)
            x = x + cond
            x = ttb(x.permute(0, 2, 1, 3).reshape(B * 16, T, -1)
                    ).reshape(B, 16, T, -1).permute(0, 2, 1, 3)
        return self.out(x[:, -1])                            # (B,16,3) delta


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class T27Full(nn.Module):
    def __init__(self, head, arm, noise_sigma=0.0):
        super().__init__()
        self.head = head
        self.arm = arm
        self.noise_sigma = noise_sigma          # mm, train-time input noise
        self.frozen = arm in ("Ffrozen", "Ffrozencoord")
        self.temporal = TemporalMixSTE(
            use_q=arm not in ("Fcoord", "Ffrozencoord"))
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

    def forward(self, b):
        if self.frozen:
            with torch.no_grad():
                refined, q = self.stage2(b)
        else:
            refined, q = self.stage2(b)
        anchor = refined
        if self.training and self.noise_sigma > 0:
            # deployed-STF2-style train-time input noise (cached_noise
            # analog): noisy tokens AND noisy residual base, clean targets;
            # test-time inputs stay clean.
            refined = refined + torch.randn_like(refined) \
                * (self.noise_sigma / 1000.0)
        B = refined.shape[0]
        c2w = b["c2w"]
        rel = torch.matmul(invert_se3(
            c2w[:, -1].reshape(B, 4, 4)).unsqueeze(1), c2w)    # (B,T,4,4)
        m = b["mask"].reshape(B, T, 1, 1) > 0
        rel = torch.where(m & m[:, -1:], rel,
                          torch.eye(4, device=rel.device).expand_as(rel))
        rel9 = torch.cat([rel[..., :3, :2].reshape(B, T, 6),
                          rel[..., :3, 3]], dim=-1)
        delta = self.temporal(q, refined, rel9)
        # anchor (2nd return) stays the CLEAN refined for the aux loss
        return refined[:, -1] + delta, anchor


def run_val(model, data, device, bs):
    model.eval()
    errs, base = [], []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            ids = range(i, min(i + bs, len(data)))
            b = data.batch(list(ids), device)
            final, _ = model(b)
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
    args = ap.parse_args()
    device = "cuda"

    init_default_scope("mmpose")
    cfg = Config.fromfile(CFG)
    head = MODELS.build(cfg.model["head"])
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    head.load_state_dict({k[5:]: v for k, v in sd.items()
                          if k.startswith("head.")}, strict=True)
    model = T27Full(head.to(device), args.arm,
                    noise_sigma=args.noise_sigma).to(device)

    val = WindowData("Val")
    print(f"val windows: {len(val)}", flush=True)

    if args.gate == "identity":
        b = val.batch(list(range(64)), device)
        model.eval()
        with torch.no_grad():
            final, refined = model(b)
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
        final, refined = model(b)
        loss = (final - b["gt"][:, -1]).norm(dim=-1).mean() \
            + 0.5 * (refined - b["gt"]).norm(dim=-1).mean()
        loss.backward()
        print(f"MEM: peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"at bs={args.bs}")
        return

    train = WindowData("Train")
    print(f"train windows: {len(train)}", flush=True)
    groups = [{"params": [p for p in model.temporal.parameters()],
               "lr": args.lr_temporal}]
    s2 = [p for p in model.head.parameters() if p.requires_grad]
    if s2:
        groups.append({"params": s2, "lr": args.lr_stage2})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [6, 8], 0.5)

    tag = f"{args.arm}" + (f"_s{args.noise_sigma:g}" if args.noise_sigma else "")
    out_dir = f"work_dirs/t27_full_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    best = math.inf
    rng = np.random.default_rng(42)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(train))
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            b = train.batch(perm[i:i + args.bs].tolist(), device)
            final, refined = model(b)
            loss = (final - b["gt"][:, -1]).norm(dim=-1).mean() \
                + 0.5 * (refined - b["gt"]).norm(dim=-1).mean()
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
