"""Task 12.2 — Ours-Ft: per-joint feature-history tokens into stage 2 (new
module file; no tracked mmpose files are edited — registered via
custom_imports).

Premise from Tasks 9/10/11: injected information must couple to the image
evidence in the FEATURE domain. Stage 2 already samples joint-adjacent
backbone features per frame; here the temporal module consumes the HISTORY
of exactly those per-joint feature vectors — the same modality stage 2
natively consumes (fixing Ours-T's coupling failure), with no warp (Task
11's gate: ego 2D moves only ~12 px median between frames).

History source: the frozen backbone/stage-1's own per-frame sampled features
(activation cache built by ours/frame_adapt/cache_ours_feats.py, keyed by
frame id; deterministic; Val cache = the frozen model's own Val outputs ->
exact test-time semantics). Loaded lazily via np.load(mmap_mode='r') slices
in get_data_info — the dataset never materializes whole sessions.

MANDATORY CONTROLS (the Task 10 lesson), selected by `history_mode`:
    "features"  — the experiment: per-joint sampled-feature history
    "replicate" — current frame's features repeated H times (separates
                  temporal information from extra parameters/capacity)
    "coords"    — 2D coordinate history through the same shell (separates
                  feature content from any history)
The claim survives only if features > replicate AND features > coords.

Head subclasses OursTemporalCascadedHead: refine()/decode()/loss()/the
zero-padded stage-2 w1 load hook are reused verbatim; only the temporal
branch input changes (`zero_f_temp` keeps the digit-for-digit baseline
guarantee, GATE 12b).

Codebase rules honored: .reshape() only, new files only.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmpose.codecs.custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)
from mmpose.registry import DATASETS, KEYPOINT_CODECS, MODELS

from my_code.custom_config.ours_t_modules import (  # noqa: F401 (registers FreezeStage1Hook)
    FreezeStage1Hook, OursTemporalCascadedHead)


# ---------------------------------------------------------------------------
# Dataset (lazy mmap slices — history windows are read per sample)
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgoposeFtDataset(KinectEgoposeDataset):
    """Paper loader + lazily-sliced per-joint feature history.

    Per-sample fields attached in get_data_info (packed by OursFtCodec):
        temporal_feats  (1,H,16,2048) sampled-feature history (fp32)
        temporal_coords (1,H,16,2)    soft-argmax coord history ([0,1])
        temporal_mask   (1,)          1.0 iff full history + cache available
    """

    def __init__(self, *, frame_export_root: str,
                 feat_cache_name: str = "feat_ours_pilot",
                 history: int = 10, **kwargs):
        self.frame_export_root = Path(frame_export_root)
        self.feat_cache_name = feat_cache_name
        self.history = int(history)
        self._mmaps = {}
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        cache_T = {}
        n_masked = 0
        H = self.history

        for d in data_list:
            img = d["img_path"]
            m = pat.search(img)
            assert m, img
            fid = int(m.group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            cd = (self.frame_export_root / participant / "actions" / session
                  / "cache" / self.feat_cache_name)
            key = str(cd)
            if key not in cache_T:
                if (cd / "feats.npy").is_file() and (cd / "coords.npy").is_file():
                    cache_T[key] = int(np.load(cd / "have.npy").shape[0])
                else:
                    cache_T[key] = -1
            T = cache_T[key]
            ok = 0 <= fid - H + 1 and fid < T
            d["_ft_dir"] = key
            d["_ft_fid"] = fid
            d["_ft_ok"] = bool(ok)
            if not ok:
                n_masked += 1
        print(f"[KinectEgoposeFtDataset] {len(data_list)} samples, "
              f"{n_masked} masked (warm-up / uncached), history={H}, "
              f"cache={self.feat_cache_name}")
        return data_list

    def _mm(self, cache_dir, name):
        key = (cache_dir, name)
        if key not in self._mmaps:
            self._mmaps[key] = np.load(
                os.path.join(cache_dir, name + ".npy"), mmap_mode="r")
        return self._mmaps[key]

    def get_data_info(self, idx):
        info = super().get_data_info(idx)
        H = self.history
        if info.get("_ft_ok"):
            fid = info["_ft_fid"]
            sl = slice(fid - H + 1, fid + 1)
            feats = self._mm(info["_ft_dir"], "feats")[sl]
            coords = self._mm(info["_ft_dir"], "coords")[sl]
            info["temporal_feats"] = np.asarray(
                feats, dtype=np.float32)[None]
            info["temporal_coords"] = np.asarray(
                coords, dtype=np.float32)[None]
            info["temporal_mask"] = np.ones(1, dtype=np.float32)
        else:
            info["temporal_feats"] = np.zeros(
                (1, H, 16, 2048), dtype=np.float32)
            info["temporal_coords"] = np.zeros(
                (1, H, 16, 2), dtype=np.float32)
            info["temporal_mask"] = np.zeros(1, dtype=np.float32)
        return info


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

@KEYPOINT_CODECS.register_module()
class OursFtCodec(Custom_mo2cap2_MSRAHeatmap):
    label_mapping_table = dict(
        Custom_mo2cap2_MSRAHeatmap.label_mapping_table,
        temporal_feats="temporal_feats",
        temporal_coords="temporal_coords",
        temporal_mask="temporal_mask",
    )


# ---------------------------------------------------------------------------
# Temporal branch (same shell as Ours-T: embed 512, last-token pooling)
# ---------------------------------------------------------------------------

class FeatureHistoryTransformer(nn.Module):
    def __init__(self, num_keypoints=16, feat_dim=2048, joint_proj_dim=32,
                 coord_mode=False, history=10, embed_dim=512, num_heads=32,
                 num_layers=8, dropout=0.1, out_dim=64):
        super().__init__()
        self.coord_mode = coord_mode
        if coord_mode:
            in_dim = num_keypoints * 2
            self.joint_proj = None
        else:
            self.joint_proj = nn.Linear(feat_dim, joint_proj_dim)
            in_dim = num_keypoints * joint_proj_dim
        self.embedding = nn.Linear(in_dim, embed_dim)
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

    def forward(self, tokens):
        """tokens: (B,H,16,C) with C = 2048 (features) or 2 (coords)."""
        B, H, J, C = tokens.shape
        if self.coord_mode:
            x = tokens.reshape(B, H, J * C)
        else:
            x = self.joint_proj(tokens).reshape(B, H, -1)
        x = self.embedding(x)
        x = x + self.pos_enc[:, :H]
        x = self.encoder(x)
        return self.out_proj(x[:, -1])


# ---------------------------------------------------------------------------
# Head (Ours-T head with the temporal input swapped)
# ---------------------------------------------------------------------------

@MODELS.register_module()
class OursFtCascadedHead(OursTemporalCascadedHead):
    """refine()/decode()/loss() and the zero-padded stage-2 w1 hook are
    inherited from OursTemporalCascadedHead; only the temporal branch and
    its input differ."""

    def __init__(self, *args, history_mode: str = "features",
                 joint_proj_dim: int = 32, f_temp_dim: int = 64,
                 history: int = 10, temporal_layers: int = 8,
                 temporal_heads: int = 32, temporal_embed: int = 512,
                 **kwargs):
        super().__init__(*args, f_temp_dim=f_temp_dim, history=history,
                         temporal_layers=temporal_layers,
                         temporal_heads=temporal_heads,
                         temporal_embed=temporal_embed, **kwargs)
        assert history_mode in ("features", "replicate", "coords"), history_mode
        self.history_mode = history_mode
        # replace the ray transformer with the feature-history shell
        self.temporal_transformer = FeatureHistoryTransformer(
            num_keypoints=self.out_channels, feat_dim=2048,
            joint_proj_dim=joint_proj_dim,
            coord_mode=(history_mode == "coords"), history=history,
            embed_dim=temporal_embed, num_heads=temporal_heads,
            num_layers=temporal_layers, out_dim=f_temp_dim)

    def _compute_f_temp(self, batch_data_samples, device):
        labels = [d.gt_instance_labels for d in batch_data_samples]
        mask = torch.cat([l.temporal_mask for l in labels]).to(device).float()
        if self.history_mode == "coords":
            tokens = torch.cat(
                [l.temporal_coords for l in labels]).to(device).float()
        else:
            tokens = torch.cat(
                [l.temporal_feats for l in labels]).to(device).float()
            if self.history_mode == "replicate":
                # capacity control: current frame's features repeated H times
                tokens = tokens[:, -1:].expand(-1, tokens.shape[1], -1, -1)
        f = self.temporal_transformer(tokens)
        f = f * mask.reshape(-1, 1)
        if self.zero_f_temp:
            f = f * 0.0
        return f
