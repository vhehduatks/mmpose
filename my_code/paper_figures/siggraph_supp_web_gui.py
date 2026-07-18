"""
SIGGRAPH supplementary-video curation GUI.

Side-by-side preview of: input image | image-only baseline | LHF-only |
LHF+GBH (ours), with per-frame MPJPE, frame scrubber, range selection, and
per-panel MP4 export. Used to curate clips for slides 5 and 6 of the supp
video.

Reuses skeleton/colors/inference helpers from generate_fig7_qualitative.

Usage:
    python my_code/paper_figures/siggraph_supp_web_gui.py
    python my_code/paper_figures/siggraph_supp_web_gui.py --port 7862 --device cuda:0

    # Skip a model at startup (saves GPU memory / load time):
    python ... --no-lhf-only
"""

import argparse
import base64
import io
import json as _json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_fig7_qualitative import (
    REPO_ROOT, DATA_ROOT,
    SKELETON, UPPER_INDICES, LOWER_INDICES,
    LINK_COLORS, KPT_COLORS,
    load_all_frames, load_model, build_pipeline, load_csv,
    run_inference, compute_mpjpe,
)

from flask import Flask, jsonify, request, Response, send_from_directory
from werkzeug.utils import secure_filename


# ── Model registry ───────────────────────────────────────────────────────
#
# Each entry's `config` and `ckpt` may be:
#   - a RELATIVE path → resolved against REPO_ROOT
#   - an ABSOLUTE path → used as-is (handy for checkpoints on a separate
#     volume, e.g. /mnt/dataset_vol/work_dir_260408/...)
# `_resolve_path()` does the discrimination at load time. The CLI flags
# `--model-config <key>=<path>` and `--model-ckpt <key>=<path>` (repeatable)
# overwrite these strings before resolution, so you can swap a model's
# paths without editing this file.

MODELS = {
    'image_only': dict(
        config='work_dirs/HMD_kinect_v5_flag_cascaded_stage1_only_image_only_10ep/HMD_kinect_v5_flag_cascaded_stage1_only_image_only_10ep_config.py',
        ckpt='work_dirs/HMD_kinect_v5_flag_cascaded_stage1_only_image_only_10ep/best_xregopose_Full Body_All_mpjpe_epoch_10.pth',
        hmd=None,              # vision-only: trained with use_hmd=False (all-zero HMD).
        hmd_dim=9,             # tensor dim still required by pipeline.
        color_scheme='baseline',
        title='Image-only',
        suffix='imageonly',
    ),
    'lhf_only': dict(
        config='work_dirs/HMD_kinect_v5_flag_cascaded_lhf_only_run3_10ep/HMD_kinect_v5_flag_cascaded_lhf_only_run3_10ep_config.py',
        ckpt='work_dirs/HMD_kinect_v5_flag_cascaded_lhf_only_run3_10ep/best_xregopose_Full Body_All_mpjpe_epoch_7.pth',
        hmd='hmd_9',
        hmd_dim=9,
        color_scheme='standard',
        title='LHF-only',
        suffix='lhfonly',
    ),
    'lhf_gbh': dict(
        config='my_code/custom_config/HMD_kinect_v5_flag_cascaded_lhf_gbh_10ep_config.py',
        ckpt='work_dirs/HMD_kinect_v5_flag_cascaded_lhf_gbh_10ep/best_xregopose_Full Body_All_mpjpe_epoch_10.pth',
        hmd='hmd_12',
        hmd_dim=12,
        color_scheme='standard',
        title='LHF+GBH (ours)',
        suffix='ours',
    ),
}

MODEL_ORDER = ['image_only', 'lhf_only', 'lhf_gbh']


def _resolve_path(p):
    """Return an absolute Path. Relative inputs are anchored on REPO_ROOT.

    Accepts str or pathlib.Path. Tilde (~) is expanded. No existence check.
    """
    p = os.path.expanduser(str(p))
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()

BASELINE_BONE = (0.902, 0.361, 0.361)   # #E65C5C
BASELINE_JOINT = (0.902, 0.361, 0.361)


def _color_scheme(name):
    if name == 'baseline':
        return [BASELINE_BONE] * len(SKELETON), [BASELINE_JOINT] * 16
    return LINK_COLORS, KPT_COLORS


# Virtual (non-inference) panels — rendered directly from frame data, no model.
#  - depth: depth PNG paired to each input image (in ego_dataset/depths/).
#  - gt:    3D ground-truth pose already loaded into frame['p3d'].
#  - gt_hmd: raw head/left/right HMD points from synced_data.csv (Unity coords,
#            no transform applied — "as-is" per user spec).

VIRTUAL_PANELS = {
    'depth':  dict(title='Depth',     suffix='depth',  kind='image'),
    'gt':     dict(title='GT',        suffix='gt',     kind='pose'),
    'gt_hmd': dict(title='GT (HMD)',  suffix='gthmd',  kind='hmd'),
}
VIRTUAL_ORDER = ['depth', 'gt', 'gt_hmd']


# ── Global state ─────────────────────────────────────────────────────────

g_sessions = []
g_session_frames = {}
g_models = {}              # {model_key: nn.Module}
g_pipeline = None
g_device = 'cuda:0'
g_cache = {}               # {(session, frame_id): {model_key: dict(pred, full, upper, lower)}}
g_selected = []            # list of selection dicts
g_video_lock = threading.Lock()

# Lazily-populated per-session caches for the virtual panels.
g_session_csv = {}         # {session: {frame_id: csv_row_dict}}

VIDEO_DIR = str(REPO_ROOT / 'my_code/my_paper/videos/siggraph_supp')
SELECTIONS_PATH = os.path.join(VIDEO_DIR, 'selections.json')


# ── Inference dispatch ───────────────────────────────────────────────────

def _hmd_for_model(frame, meta):
    """Return the HMD vector to feed `meta`'s model.

    - If meta['hmd'] is None → vision-only training (use_hmd=False), so the
      model only ever saw all-zero HMD; pass zeros of meta['hmd_dim'].
    - Otherwise pass the named field from the frame, defaulting to zeros if
      the frame is missing it.
    """
    field = meta.get('hmd')
    dim = meta.get('hmd_dim', 9)
    if field is None:
        return np.zeros(dim, dtype=np.float32)
    hmd = frame.get(field)
    if hmd is None:
        return np.zeros(dim, dtype=np.float32)
    return hmd


def get_predictions(session, frame_idx, model_keys):
    frame = g_session_frames[session][frame_idx]
    cache_entry = g_cache.setdefault((session, frame['frame_id']), {})
    for k in model_keys:
        if k in cache_entry or k not in g_models:
            continue
        meta = MODELS[k]
        hmd = _hmd_for_model(frame, meta)
        pred = run_inference(g_models[k], g_pipeline, frame, hmd, g_device)
        f, u, l = compute_mpjpe(pred, frame['p3d'])
        cache_entry[k] = dict(pred=pred, full=float(f), upper=float(u), lower=float(l))
    return cache_entry


# ── Virtual-panel helpers (depth / GT / GT-HMD) ──────────────────────────

def _depth_path(img_path):
    """Map .../ego_dataset/images/frame_NNNNNN.jpg → .../depths/frame_NNNNNN.png."""
    p = Path(img_path)
    name = p.stem + '.png'
    return str(p.parent.parent / 'depths' / name)


def _session_csv(session):
    """Return cached csv lookup for a session, loading on first access."""
    if session in g_session_csv:
        return g_session_csv[session]
    csv_path = Path(DATA_ROOT) / session / 'synced_data.csv'
    try:
        g_session_csv[session] = load_csv(str(csv_path)) if csv_path.is_file() else {}
    except Exception:
        g_session_csv[session] = {}
    return g_session_csv[session]


def _raw_hmd_points(session, frame_id):
    """Return (head, left, right) as raw Unity (x,y,z) — no transform.
    Returns None if unavailable.
    """
    rows = _session_csv(session)
    row = rows.get(int(frame_id))
    if row is None:
        return None
    try:
        head = np.array([float(row['hmd_pos_x']),
                         float(row['hmd_pos_y']),
                         float(row['hmd_pos_z'])], dtype=np.float32)
        left = np.array([float(row['left_pos_x']),
                         float(row['left_pos_y']),
                         float(row['left_pos_z'])], dtype=np.float32)
        right = np.array([float(row['right_pos_x']),
                          float(row['right_pos_y']),
                          float(row['right_pos_z'])], dtype=np.float32)
    except (KeyError, ValueError):
        return None
    return head, left, right


def _read_depth_visu(depth_path, panel_size=None):
    """Load a depth PNG and return a viewable BGR image. None if unreadable."""
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        # Already a colorized depth — keep as-is.
        vis = raw[..., :3]
    else:
        d = raw.astype(np.float32)
        valid = d > 0
        if valid.any():
            lo, hi = np.percentile(d[valid], (2, 98))
            if hi <= lo:
                hi = lo + 1.0
            d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        else:
            d = np.zeros_like(d)
        d8 = (d * 255).astype(np.uint8)
        vis = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    if panel_size is not None and vis.shape[:2] != (panel_size, panel_size):
        vis = cv2.resize(vis, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    return vis


# ── Skeleton rendering (mirrors stage_comparison_web_gui style) ──────────

def _draw_skeleton(ax, pose, link_colors, kpt_colors, lw, kpt_s,
                   alpha, ls, edge_c, edge_w, zorder):
    for idx, (i, j) in enumerate(SKELETON):
        pts = pose[[i, j]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=link_colors[idx], linewidth=lw, alpha=alpha,
                linestyle=ls)
    for k in range(16):
        ax.scatter(pose[k, 0], pose[k, 1], pose[k, 2],
                   c=[kpt_colors[k]], s=kpt_s, alpha=alpha,
                   edgecolors=edge_c, linewidths=edge_w, zorder=zorder)


def _render_pose_b64(gt, pred, scheme, elev, azim, show_gt, show_pred,
                     line_width, zoom, hide_axes, title=''):
    fig = plt.figure(figsize=(4, 3.8), dpi=110)
    ax = fig.add_subplot(111, projection='3d')

    pts_all = []
    if show_gt and gt is not None:
        pts_all.append(gt)
    if show_pred and pred is not None:
        pts_all.append(pred)
    if not pts_all:
        plt.close(fig)
        return ''

    combined = np.concatenate(pts_all, axis=0)
    center = combined.mean(axis=0)
    rng = max(np.abs(combined - center).max(), 0.3) * 1.3 * zoom

    gt_lw = max(line_width * 0.8, 1.0)
    ks_p = max(16 * line_width, 20)
    ks_g = max(12 * line_width, 15)

    if show_gt and gt is not None:
        _draw_skeleton(ax, gt,
                       [(0.55, 0.55, 0.55)] * len(SKELETON),
                       [(0.45, 0.45, 0.45)] * 16,
                       gt_lw, ks_g, 0.5, '--', 'gray', 0.3, 3)
    if show_pred and pred is not None:
        link_c, kpt_c = _color_scheme(scheme)
        _draw_skeleton(ax, pred, link_c, kpt_c,
                       line_width, ks_p, 1.0, '-', 'white', 0.5, 5)

    ax.set_xlim(center[0] - rng, center[0] + rng)
    ax.set_ylim(center[1] - rng, center[1] + rng)
    ax.set_zlim(center[2] - rng, center[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    for fn in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        fn([])
    ax.tick_params(axis='both', which='both', length=0)
    if hide_axes:
        ax.set_axis_off()
        fig.patch.set_alpha(0.0)
    if title:
        ax.set_title(title, fontsize=10, pad=4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                transparent=hide_axes)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_input_b64(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return ''
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


def _render_depth_b64(depth_path):
    vis = _read_depth_visu(depth_path)
    if vis is None:
        return ''
    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


def _render_gt_hmd_b64(points, elev, azim, line_width, zoom, hide_axes, title=''):
    """Render the 3 raw HMD markers (head/left/right) as a sparse skeleton."""
    if points is None:
        return ''
    head, left, right = points
    fig = plt.figure(figsize=(4, 3.8), dpi=110)
    ax = fig.add_subplot(111, projection='3d')

    pts = np.stack([head, left, right], axis=0)
    center = pts.mean(axis=0)
    rng = max(np.abs(pts - center).max(), 0.3) * 1.4 * zoom

    # head→hand bones
    for hand, color in ((left, (0.20, 0.55, 0.85)), (right, (0.85, 0.40, 0.20))):
        ax.plot([head[0], hand[0]], [head[1], hand[1]], [head[2], hand[2]],
                color=color, linewidth=max(line_width, 1.5), alpha=0.95)
    ax.scatter(*head, c=[(0.10, 0.10, 0.10)],
               s=max(28 * line_width, 50), edgecolors='white', linewidths=0.6, zorder=6)
    ax.scatter(*left, c=[(0.20, 0.55, 0.85)],
               s=max(22 * line_width, 36), edgecolors='white', linewidths=0.6, zorder=6)
    ax.scatter(*right, c=[(0.85, 0.40, 0.20)],
               s=max(22 * line_width, 36), edgecolors='white', linewidths=0.6, zorder=6)

    ax.set_xlim(center[0] - rng, center[0] + rng)
    ax.set_ylim(center[1] - rng, center[1] + rng)
    ax.set_zlim(center[2] - rng, center[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    for fn in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        fn([])
    ax.tick_params(axis='both', which='both', length=0)
    if hide_axes:
        ax.set_axis_off()
        fig.patch.set_alpha(0.0)
    if title:
        ax.set_title(title, fontsize=10, pad=4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                transparent=hide_axes)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Video panel rendering (numpy BGR, fixed size) ────────────────────────

def _render_pose_np(gt, pred, scheme, elev, azim, show_gt, show_pred,
                    line_width, zoom, hide_axes, width, height, dpi):
    figsize_in = 5.0
    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(111, projection='3d')

    pts_all = []
    if show_gt and gt is not None:
        pts_all.append(gt)
    if show_pred and pred is not None:
        pts_all.append(pred)
    if not pts_all:
        plt.close(fig)
        return np.full((height, width, 3), 255, dtype=np.uint8)

    combined = np.concatenate(pts_all, axis=0)
    center = combined.mean(axis=0)
    rng = max(np.abs(combined - center).max(), 0.3) * 1.3 * zoom

    gt_lw = max(line_width * 0.8, 1.0)
    ks_p = max(16 * line_width, 20)
    ks_g = max(12 * line_width, 15)

    if show_gt and gt is not None:
        _draw_skeleton(ax, gt,
                       [(0.55, 0.55, 0.55)] * len(SKELETON),
                       [(0.45, 0.45, 0.45)] * 16,
                       gt_lw, ks_g, 0.5, '--', 'gray', 0.3, 3)
    if show_pred and pred is not None:
        link_c, kpt_c = _color_scheme(scheme)
        _draw_skeleton(ax, pred, link_c, kpt_c,
                       line_width, ks_p, 1.0, '-', 'white', 0.5, 5)

    ax.set_xlim(center[0] - rng, center[0] + rng)
    ax.set_ylim(center[1] - rng, center[1] + rng)
    ax.set_zlim(center[2] - rng, center[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    for fn in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        fn([])
    ax.tick_params(axis='both', which='both', length=0)
    if hide_axes:
        ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def _label_bar(width, title, sub, bar_h=34):
    bar = np.full((bar_h, width, 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def _input_panel_np(frame, panel_size, show_labels):
    inp = cv2.imread(frame['img_path'])
    if inp is None:
        inp = np.full((panel_size, panel_size, 3), 255, dtype=np.uint8)
    else:
        inp = cv2.resize(inp, (panel_size, panel_size),
                         interpolation=cv2.INTER_AREA)
    if show_labels:
        sub = f"{frame['action']}  frame {int(frame['frame_id']):d}"
        inp = np.vstack([_label_bar(panel_size, 'Input', sub), inp])
    return inp


def _depth_panel_np(frame, panel_size, show_labels):
    vis = _read_depth_visu(_depth_path(frame['img_path']), panel_size=panel_size)
    if vis is None:
        vis = np.full((panel_size, panel_size, 3), 0, dtype=np.uint8)
        cv2.putText(vis, 'depth missing', (10, panel_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    if show_labels:
        sub = f"{frame['action']}  frame {int(frame['frame_id']):d}"
        vis = np.vstack([_label_bar(panel_size, 'Depth', sub), vis])
    return vis


def _gt_hmd_panel_np(session, frame, panel_size, dpi,
                     elev, azim, line_width, zoom, hide_axes,
                     show_labels):
    points = _raw_hmd_points(session, frame['frame_id'])
    width = height = panel_size
    if points is None:
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(img, 'no csv row', (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    else:
        head, left, right = points
        figsize_in = 5.0
        fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax = fig.add_subplot(111, projection='3d')
        pts = np.stack([head, left, right], axis=0)
        center = pts.mean(axis=0)
        rng = max(np.abs(pts - center).max(), 0.3) * 1.4 * zoom
        for hand, color in ((left, (0.20, 0.55, 0.85)), (right, (0.85, 0.40, 0.20))):
            ax.plot([head[0], hand[0]], [head[1], hand[1]], [head[2], hand[2]],
                    color=color, linewidth=max(line_width, 1.5), alpha=0.95)
        ax.scatter(*head, c=[(0.10, 0.10, 0.10)],
                   s=max(28 * line_width, 50), edgecolors='white', linewidths=0.6, zorder=6)
        ax.scatter(*left, c=[(0.20, 0.55, 0.85)],
                   s=max(22 * line_width, 36), edgecolors='white', linewidths=0.6, zorder=6)
        ax.scatter(*right, c=[(0.85, 0.40, 0.20)],
                   s=max(22 * line_width, 36), edgecolors='white', linewidths=0.6, zorder=6)
        ax.set_xlim(center[0] - rng, center[0] + rng)
        ax.set_ylim(center[1] - rng, center[1] + rng)
        ax.set_zlim(center[2] - rng, center[2] + rng)
        ax.view_init(elev=elev, azim=azim)
        for fn in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            fn([])
        ax.tick_params(axis='both', which='both', length=0)
        if hide_axes:
            ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((height, width, 3), 255, dtype=np.uint8)
        elif img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if show_labels:
        sub = f"{frame['action']}  frame {int(frame['frame_id']):d}"
        img = np.vstack([_label_bar(panel_size, 'GT (HMD)', sub), img])
    return img


def _gt_panel_np(frame, panel_size, dpi,
                 elev, azim, line_width, zoom, hide_axes, show_labels):
    # Render GT through the `pred` slot so it gets the vivid color scheme
    # (the `gt` slot in _render_pose_np always draws gray ghost lines).
    img = _render_pose_np(None, frame['p3d'], 'standard',
                          elev, azim, False, True,
                          line_width, zoom, hide_axes,
                          panel_size, panel_size, dpi)
    if show_labels:
        sub = f"{frame['action']}  frame {int(frame['frame_id']):d}"
        img = np.vstack([_label_bar(panel_size, 'GT', sub), img])
    return img


def _pose_panel_np(gt, pred, scheme, elev, azim, show_gt, show_pred,
                   line_width, zoom, hide_axes, panel_size, dpi,
                   title, sub_label, show_labels):
    img = _render_pose_np(gt, pred, scheme, elev, azim, show_gt, show_pred,
                          line_width, zoom, hide_axes,
                          panel_size, panel_size, dpi)
    if show_labels:
        img = np.vstack([_label_bar(panel_size, title, sub_label), img])
    return img


def _strip_metadata(path):
    """Best-effort metadata strip via exiftool. Silently no-op if absent."""
    if not shutil.which('exiftool'):
        return False
    try:
        subprocess.run(['exiftool', '-overwrite_original', '-all=', path],
                       check=False, capture_output=True, timeout=30)
        return True
    except Exception:
        return False


# ── Flask app ────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/api/sessions')
def api_sessions():
    out = []
    for s in g_sessions:
        action = '_'.join(s.split('_')[:-2])
        n = len(g_session_frames[s])
        out.append(dict(name=s, action=action, n_frames=n))
    return jsonify(out)


@app.route('/api/models')
def api_models():
    out = []
    for k in MODEL_ORDER:
        if k not in g_models:
            continue
        m = MODELS[k]
        out.append(dict(key=k, title=m['title'], suffix=m['suffix'],
                        color_scheme=m['color_scheme']))
    return jsonify(out)


@app.route('/api/virtuals')
def api_virtuals():
    """Non-inference panels (depth / GT / GT-HMD) the frontend should expose."""
    out = []
    for k in VIRTUAL_ORDER:
        m = VIRTUAL_PANELS[k]
        out.append(dict(key=k, title=m['title'], suffix=m['suffix'], kind=m['kind']))
    return jsonify(out)


@app.route('/api/render')
def api_render():
    session = request.args.get('session', g_sessions[0] if g_sessions else '')
    frame_idx = int(request.args.get('frame', 0))
    azim = int(request.args.get('azim', 70))
    elev = int(request.args.get('elev', 15))
    show_gt = request.args.get('show_gt', 'true') == 'true'
    show_pred = request.args.get('show_pred', 'true') == 'true'
    lw = float(request.args.get('line_width', 2.5))
    zoom = float(request.args.get('zoom', 1.0))
    hide = request.args.get('hide_axes', 'false') == 'true'
    raw_keys = [k for k in request.args.get('models', '').split(',') if k]
    model_keys = [k for k in raw_keys if k in g_models]
    virtual_keys = [k for k in raw_keys if k in VIRTUAL_PANELS]
    if not model_keys and not virtual_keys and not raw_keys:
        model_keys = list(g_models.keys())

    if session not in g_session_frames:
        return jsonify(error='session not found'), 404
    frames = g_session_frames[session]
    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    frame = frames[frame_idx]

    cache_entry = get_predictions(session, frame_idx, model_keys)
    img_b64 = _render_input_b64(frame['img_path'])

    panels = {}
    mpjpe = {}
    for k in model_keys:
        if k not in cache_entry:
            continue
        e = cache_entry[k]
        title = (f"{MODELS[k]['title']}  full={e['full']:.0f}mm  "
                 f"L={e['lower']:.0f}mm")
        panels[k] = _render_pose_b64(
            frame['p3d'] if show_gt else None,
            e['pred'] if show_pred else None,
            MODELS[k]['color_scheme'], elev, azim,
            show_gt, show_pred, lw, zoom, hide, title=title)
        mpjpe[k] = dict(full=round(e['full'], 1),
                        upper=round(e['upper'], 1),
                        lower=round(e['lower'], 1))

    # Virtual panels: rendered directly from frame data, no inference.
    depth_b64 = ''
    for k in virtual_keys:
        meta = VIRTUAL_PANELS[k]
        if meta['kind'] == 'image':
            # 'depth' returns a separate image block (not a 3D matplotlib panel)
            depth_b64 = _render_depth_b64(_depth_path(frame['img_path']))
        elif meta['kind'] == 'pose':
            # GT pose — render through `pred` slot so colors are vivid.
            panels[k] = _render_pose_b64(
                None, frame['p3d'], 'standard', elev, azim,
                False, True, lw, zoom, hide, title=meta['title'])
        elif meta['kind'] == 'hmd':
            panels[k] = _render_gt_hmd_b64(
                _raw_hmd_points(session, frame['frame_id']),
                elev, azim, lw, zoom, hide, title=meta['title'])

    return jsonify(
        img=img_b64,
        depth=depth_b64,
        panels=panels,
        mpjpe=mpjpe,
        action=frame['action'],
        session=session,
        frame_id=int(frame['frame_id']),
        n_frames=len(frames),
    )


# ── Selections ───────────────────────────────────────────────────────────

@app.route('/api/sel_add', methods=['POST'])
def api_sel_add():
    body = request.get_json(force=True)
    sess = body['session']
    if sess not in g_session_frames:
        return jsonify(error='session not found'), 404
    n = len(g_session_frames[sess])
    start = max(0, min(int(body['start']), n - 1))
    end = max(start, min(int(body['end']), n - 1))
    label = body.get('label', '').strip() or '_'.join(sess.split('_')[:-2])
    requested = [k for k in body.get('models', [])
                 if k in g_models or k in VIRTUAL_PANELS]
    if not requested:
        requested = list(g_models.keys())
    # Preserve order, dedupe, ensure 'input' is first.
    seen = set()
    ordered = []
    for k in ['input'] + requested:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    g_selected.append(dict(
        action=label,
        sequence_id=sess,
        start_frame=start,
        end_frame=end,
        models=ordered,
    ))
    return jsonify(selected=g_selected)


@app.route('/api/sel_remove', methods=['POST'])
def api_sel_remove():
    body = request.get_json(force=True)
    idx = int(body['index'])
    if 0 <= idx < len(g_selected):
        g_selected.pop(idx)
    return jsonify(selected=g_selected)


@app.route('/api/sel_clear', methods=['POST'])
def api_sel_clear():
    g_selected.clear()
    return jsonify(selected=g_selected)


@app.route('/api/sel_save', methods=['POST'])
def api_sel_save():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    with open(SELECTIONS_PATH, 'w') as f:
        _json.dump(g_selected, f, indent=2)
    return jsonify(path=SELECTIONS_PATH, count=len(g_selected))


@app.route('/api/sel_load', methods=['POST'])
def api_sel_load():
    if not os.path.exists(SELECTIONS_PATH):
        return jsonify(error='no selections.json yet', path=SELECTIONS_PATH), 404
    with open(SELECTIONS_PATH) as f:
        loaded = _json.load(f)
    g_selected.clear()
    g_selected.extend(loaded)
    return jsonify(selected=g_selected, path=SELECTIONS_PATH)


# ── Scan: per-session mean MPJPE across toggled models ──────────────────

@app.route('/api/scan')
def api_scan():
    """Rank sessions by mean MPJPE across the user-toggled models.

    Comparison metric depends on which toggled models are present:
    - lhf_gbh + at least one other → metric = min(other_means) − ours_mean
      (largest positive value = ours wins by most on average)
    - lhf_gbh missing, ≥2 toggled → metric = max(means) − min(means)
      (largest spread = sessions where toggled models disagree most)
    - exactly 1 toggled → metric = -mean (rank by lowest mean MPJPE = best)

    `threshold` filters: keep only sessions where metric ≥ threshold (mm).
    """
    threshold = float(request.args.get('threshold', 5))
    target = request.args.get('session', '__all__')
    sess_list = g_sessions if target == '__all__' else [target]
    requested = [k for k in request.args.get('models', '').split(',') if k]
    requested = [k for k in requested if k in g_models]
    if not requested:
        return Response(
            f"data: {_json.dumps({'type':'error','msg':'No toggled models loaded — toggle at least one model column.'})}\n\n",
            mimetype='text/event-stream')

    total = sum(len(g_session_frames.get(s, [])) for s in sess_list)

    def generate():
        results = []
        done = 0
        for sess in sess_list:
            if sess not in g_session_frames:
                continue
            frames = g_session_frames[sess]
            sums = {k: dict(full=0.0, upper=0.0, lower=0.0) for k in requested}
            for fi, _ in enumerate(frames):
                ce = get_predictions(sess, fi, requested)
                for k in requested:
                    if k not in ce:
                        continue
                    sums[k]['full']  += ce[k]['full']
                    sums[k]['upper'] += ce[k]['upper']
                    sums[k]['lower'] += ce[k]['lower']
                done += 1
                if done % 20 == 0 or done == total:
                    yield f"data: {_json.dumps({'type':'progress','done':done,'total':total,'sessions':len(results)})}\n\n"

            n = max(1, len(frames))
            means = {k: {p: sums[k][p] / n for p in ('full', 'upper', 'lower')}
                     for k in requested}
            full_means = [means[k]['full'] for k in requested]

            if 'lhf_gbh' in requested and len(requested) >= 2:
                others_full = [means[k]['full'] for k in requested if k != 'lhf_gbh']
                others_lo = [means[k]['lower'] for k in requested if k != 'lhf_gbh']
                metric = min(others_full) - means['lhf_gbh']['full']
                metric_lo = min(others_lo) - means['lhf_gbh']['lower']
            elif len(requested) >= 2:
                metric = max(full_means) - min(full_means)
                lo_means = [means[k]['lower'] for k in requested]
                metric_lo = max(lo_means) - min(lo_means)
            else:
                # only one toggled — sort by best (lowest) MPJPE
                metric = -full_means[0]
                metric_lo = -means[requested[0]]['lower']

            if metric < threshold:
                continue

            results.append(dict(
                session=sess,
                action='_'.join(sess.split('_')[:-2]) or sess,
                n_frames=len(frames),
                means={k: {p: round(v, 1) for p, v in means[k].items()}
                       for k in requested},
                metric=round(metric, 1),
                metric_lo=round(metric_lo, 1),
            ))

        results.sort(key=lambda x: x['metric'], reverse=True)
        yield f"data: {_json.dumps({'type':'done','results':results,'total':len(results),'models':requested})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# ── Video render ─────────────────────────────────────────────────────────

def _build_panel(k, session, frame, panel_size, dpi,
                 elev, azim, show_gt, show_pred,
                 line_width, zoom, hide_axes, show_labels, ce):
    """Build a single output panel by key. Returns BGR np array or None."""
    if k == 'input':
        return _input_panel_np(frame, panel_size, show_labels)
    if k == 'depth':
        return _depth_panel_np(frame, panel_size, show_labels)
    if k == 'gt':
        return _gt_panel_np(frame, panel_size, dpi,
                            elev, azim, line_width, zoom, hide_axes, show_labels)
    if k == 'gt_hmd':
        return _gt_hmd_panel_np(session, frame, panel_size, dpi,
                                elev, azim, line_width, zoom, hide_axes, show_labels)
    if k in g_models:
        e = ce.get(k)
        if e is None:
            return None
        sub = (f"{frame['action']}  f{int(frame['frame_id']):d}  "
               f"full={e['full']:.0f}mm  L={e['lower']:.0f}mm")
        return _pose_panel_np(
            frame['p3d'], e['pred'], MODELS[k]['color_scheme'],
            elev, azim, show_gt, show_pred,
            line_width, zoom, hide_axes,
            panel_size, dpi, MODELS[k]['title'], sub, show_labels)
    return None


def _panel_suffix(k):
    if k == 'input':
        return 'input'
    if k in VIRTUAL_PANELS:
        return VIRTUAL_PANELS[k]['suffix']
    return MODELS[k]['suffix']


def _panel_title(k):
    if k == 'input':
        return 'Input'
    if k in VIRTUAL_PANELS:
        return VIRTUAL_PANELS[k]['title']
    return MODELS[k]['title']


def _render_clip(session, start, end, model_keys, prefix, fps, panel_size,
                 azim, elev, show_gt, show_pred, line_width, zoom,
                 hide_axes, show_labels, dpi, anonymize, progress_cb):
    """Render per-panel mp4s for one selection. Returns list of file infos."""
    frames = g_session_frames[session]
    start = max(0, min(start, len(frames) - 1))
    end = max(start, min(end, len(frames) - 1))
    total = end - start + 1

    inference_keys = [k for k in model_keys if k in g_models]

    def build(idx):
        frame = frames[idx]
        ce = get_predictions(session, idx, inference_keys) if inference_keys else {}
        out = {}
        for k in model_keys:
            panel = _build_panel(k, session, frame, panel_size, dpi,
                                 elev, azim, show_gt, show_pred,
                                 line_width, zoom, hide_axes, show_labels, ce)
            if panel is not None:
                out[k] = panel
        return out

    probe = build(start)
    if not probe:
        raise RuntimeError('no panels produced (empty model selection)')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    paths = {}
    writers = {}
    for k, panel in probe.items():
        suffix = _panel_suffix(k)
        p = os.path.join(VIDEO_DIR, f'{prefix}_{suffix}.mp4')
        h, w = panel.shape[:2]
        wr = cv2.VideoWriter(p, fourcc, fps, (w, h))
        if not wr.isOpened():
            for other in writers.values():
                other.release()
            raise RuntimeError(f'VideoWriter failed to open for {k}')
        paths[k] = p
        writers[k] = wr

    try:
        for k, panel in probe.items():
            writers[k].write(panel)
        progress_cb(1, total)

        for i in range(start + 1, end + 1):
            panels = build(i)
            for k, wr in writers.items():
                if k in panels:
                    wr.write(panels[k])
            done = i - start + 1
            if done % 5 == 0 or done == total:
                progress_cb(done, total)
    finally:
        for wr in writers.values():
            wr.release()

    files = []
    for k, p in paths.items():
        if anonymize:
            _strip_metadata(p)
        try:
            size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)
        except OSError:
            size_mb = 0.0
        files.append(dict(
            key=k,
            title=_panel_title(k),
            filename=os.path.basename(p),
            path=p,
            size_mb=size_mb,
        ))
    return files


@app.route('/api/video_render')
def api_video_render():
    session = request.args.get('session', g_sessions[0] if g_sessions else '')
    try:
        start = int(request.args.get('start', 0))
        end = int(request.args.get('end', 0))
    except ValueError:
        return Response(
            f"data: {_json.dumps({'type':'error','msg':'Invalid start/end'})}\n\n",
            mimetype='text/event-stream')
    fps = float(request.args.get('fps', 30))
    panel_size = int(request.args.get('panel_size', 1080))
    azim = int(request.args.get('azim', 70))
    elev = int(request.args.get('elev', 15))
    show_gt = request.args.get('show_gt', 'false') == 'true'
    show_pred = request.args.get('show_pred', 'true') == 'true'
    line_width = float(request.args.get('line_width', 2.5))
    zoom = float(request.args.get('zoom', 1.0))
    hide_axes = request.args.get('hide_axes', 'true') == 'true'
    show_labels = request.args.get('show_labels', 'false') == 'true'
    dpi = max(50, min(int(request.args.get('dpi', 200)), 600))
    anonymize = request.args.get('anonymize', 'true') == 'true'
    raw_name = (request.args.get('filename', '') or '').strip()
    requested = [k for k in request.args.get('models', '').split(',') if k]
    if not requested:
        requested = ['input'] + list(g_models.keys())

    def err_stream(msg):
        return Response(
            f"data: {_json.dumps({'type':'error','msg':msg})}\n\n",
            mimetype='text/event-stream')

    if session not in g_session_frames:
        return err_stream('Session not found')

    n = len(g_session_frames[session])
    if n == 0:
        return err_stream('Session has no frames')

    os.makedirs(VIDEO_DIR, exist_ok=True)
    if raw_name:
        prefix = secure_filename(raw_name)
        if prefix.lower().endswith('.mp4'):
            prefix = prefix[:-4]
    else:
        action = '_'.join(session.split('_')[:-2]) or session
        prefix = secure_filename(f'{action}_{start}-{end}')
    if not prefix:
        prefix = f'video_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    def generate():
        with g_video_lock:
            try:
                last = [0]

                def progress_cb(done, total):
                    last[0] = done
                    # buffered yield happens via the outer loop;
                    # we instead push events directly here via a queue is overkill —
                    # simpler: use a closure-shared list and let the wrapper poll.
                    pass

                # Simpler: do progress streaming inline by replicating the loop.
                frames = g_session_frames[session]
                clamped_start = max(0, min(start, n - 1))
                clamped_end = max(clamped_start, min(end, n - 1))
                total = clamped_end - clamped_start + 1

                inference_keys = [k for k in requested if k in g_models]

                def build(idx):
                    frame = frames[idx]
                    ce = get_predictions(session, idx, inference_keys) if inference_keys else {}
                    out = {}
                    for k in requested:
                        panel = _build_panel(k, session, frame, panel_size, dpi,
                                             elev, azim, show_gt, show_pred,
                                             line_width, zoom, hide_axes, show_labels, ce)
                        if panel is not None:
                            out[k] = panel
                    return out

                probe = build(clamped_start)
                if not probe:
                    yield f"data: {_json.dumps({'type':'error','msg':'No panels (empty model selection)'})}\n\n"
                    return

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                paths = {}
                writers = {}
                for k, panel in probe.items():
                    suffix = _panel_suffix(k)
                    p = os.path.join(VIDEO_DIR, f'{prefix}_{suffix}.mp4')
                    h, w = panel.shape[:2]
                    wr = cv2.VideoWriter(p, fourcc, fps, (w, h))
                    if not wr.isOpened():
                        for other in writers.values():
                            other.release()
                        yield f"data: {_json.dumps({'type':'error','msg':f'VideoWriter open failed for {k}'})}\n\n"
                        return
                    paths[k] = p
                    writers[k] = wr

                try:
                    for k, panel in probe.items():
                        writers[k].write(panel)
                    yield f"data: {_json.dumps({'type':'progress','done':1,'total':total})}\n\n"

                    for i in range(clamped_start + 1, clamped_end + 1):
                        try:
                            panels = build(i)
                        except Exception as ex:
                            yield f"data: {_json.dumps({'type':'error','msg':f'Frame {i} failed: {ex}'})}\n\n"
                            return
                        for k, wr in writers.items():
                            if k in panels:
                                wr.write(panels[k])
                        done = i - clamped_start + 1
                        if done % 5 == 0 or done == total:
                            yield f"data: {_json.dumps({'type':'progress','done':done,'total':total})}\n\n"
                finally:
                    for wr in writers.values():
                        wr.release()

                files = []
                anonymized = 0
                for k, p in paths.items():
                    if anonymize and _strip_metadata(p):
                        anonymized += 1
                    try:
                        size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)
                    except OSError:
                        size_mb = 0.0
                    files.append(dict(
                        key=k,
                        title=_panel_title(k),
                        filename=os.path.basename(p),
                        path=p,
                        size_mb=size_mb,
                    ))

                yield f"data: {_json.dumps({'type':'done','files':files,'frames':total,'fps':fps,'dpi':dpi,'anonymized':anonymized})}\n\n"
            except Exception as ex:
                yield f"data: {_json.dumps({'type':'error','msg':f'Render failed: {ex}'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/render_selections')
def api_render_selections():
    """Iterate the in-memory selections list and render each clip in turn."""
    fps = float(request.args.get('fps', 30))
    panel_size = int(request.args.get('panel_size', 1080))
    azim = int(request.args.get('azim', 70))
    elev = int(request.args.get('elev', 15))
    line_width = float(request.args.get('line_width', 2.5))
    zoom = float(request.args.get('zoom', 1.0))
    hide_axes = request.args.get('hide_axes', 'true') == 'true'
    show_labels = request.args.get('show_labels', 'false') == 'true'
    dpi = max(50, min(int(request.args.get('dpi', 200)), 600))
    anonymize = request.args.get('anonymize', 'true') == 'true'

    if not g_selected:
        return Response(
            f"data: {_json.dumps({'type':'error','msg':'Selection list is empty'})}\n\n",
            mimetype='text/event-stream')

    os.makedirs(VIDEO_DIR, exist_ok=True)

    def generate():
        with g_video_lock:
            all_files = []
            for sel_idx, sel in enumerate(g_selected):
                sess = sel['sequence_id']
                if sess not in g_session_frames:
                    yield f"data: {_json.dumps({'type':'error','msg':f'[{sel_idx}] session not found: {sess}'})}\n\n"
                    continue
                start = int(sel['start_frame'])
                end = int(sel['end_frame'])
                requested = [k for k in sel.get('models', [])
                             if k == 'input' or k in g_models or k in VIRTUAL_PANELS]
                if not requested:
                    requested = ['input', 'lhf_gbh'] if 'lhf_gbh' in g_models else (['input'] + list(g_models.keys()))
                action_label = sel.get('action', '_'.join(sess.split('_')[:-2]))
                prefix = secure_filename(action_label) or f'sel{sel_idx:02d}'

                yield f"data: {_json.dumps({'type':'clip_start','index':sel_idx,'total':len(g_selected),'prefix':prefix,'frames':end-start+1})}\n\n"
                try:
                    files = _render_clip(
                        session=sess, start=start, end=end,
                        model_keys=requested, prefix=prefix,
                        fps=fps, panel_size=panel_size,
                        azim=azim, elev=elev,
                        show_gt=False, show_pred=True,
                        line_width=line_width, zoom=zoom,
                        hide_axes=hide_axes, show_labels=show_labels,
                        dpi=dpi, anonymize=anonymize,
                        progress_cb=lambda d, t: None,
                    )
                    all_files.extend(files)
                    yield f"data: {_json.dumps({'type':'clip_done','index':sel_idx,'files':files})}\n\n"
                except Exception as ex:
                    yield f"data: {_json.dumps({'type':'error','msg':f'[{sel_idx}] {ex}'})}\n\n"
            yield f"data: {_json.dumps({'type':'done','files':all_files,'count':len(all_files)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/video_file/<path:filename>')
def api_video_file(filename):
    return send_from_directory(VIDEO_DIR, filename, mimetype='video/mp4',
                               as_attachment=False)


# ── HTML ─────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIGGRAPH Supp — Sequence Curation</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #333; }
.header { background: #0A1628; color: #fff; padding: 12px 24px; }
.header h1 { font-size: 18px; }
.header p { font-size: 12px; color: #99a; margin-top: 2px; }

.controls { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 24px;
            background: #fff; border-bottom: 1px solid #ddd; align-items: end; }
.ctrl-group { display: flex; flex-direction: column; gap: 2px; }
.ctrl-group label { font-size: 11px; font-weight: 600; color: #666;
                    text-transform: uppercase; }
.ctrl-group select, .ctrl-group input[type=number],
.ctrl-group input[type=text] { padding: 4px 8px; border-radius: 4px;
                                border: 1px solid #ccc; }
.range-row { display: flex; align-items: center; gap: 6px; }
.range-row input[type=range] { width: 130px; }
.range-row .val { font-size: 12px; font-weight: 600; min-width: 32px; text-align: right; }
.chk-row { display: flex; gap: 14px; align-items: center; padding-top: 4px;
           flex-wrap: wrap; }
.chk-row label { font-size: 12px; cursor: pointer; }
.nav-row { display: flex; gap: 6px; align-items: end; }
.nav-row button { padding: 5px 14px; border: 1px solid #ccc; border-radius: 4px;
                   background: #fff; cursor: pointer; font-size: 13px; }
.nav-row button:hover { background: #eee; }
.frame-num { font-size: 13px; font-weight: 600; padding: 5px 0; }

.main { display: flex; gap: 16px; padding: 16px 24px; }
.images { flex: 3; display: flex; gap: 8px; flex-wrap: wrap; }
.images .panel { flex: 1; min-width: 240px; background: #fff; border-radius: 8px;
                  overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.images .panel img { width: 100%; display: block; }
.images .panel .label { padding: 6px 10px; font-size: 12px; font-weight: 600;
                         text-align: center; background: #f9f9f9;
                         border-top: 1px solid #eee; }
.panel.baseline .label { background: #fde2e2; color: #8a1a1a; }
.panel.ours .label { background: #e2f5e2; color: #1a5f1a; }
.panel.input .label { background: #e2eaf5; color: #1a3a6f; }

.sidebar { flex: 1; min-width: 340px; }
.card { background: #fff; border-radius: 8px; padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px; }
.card h3 { font-size: 13px; margin-bottom: 8px; }

table.mpjpe { width: 100%; border-collapse: collapse; font-size: 12px; }
table.mpjpe th, table.mpjpe td { padding: 3px 8px; text-align: right; }
table.mpjpe th { background: #f0f0f0; font-weight: 600; }
table.mpjpe td:first-child, table.mpjpe th:first-child { text-align: left; }

.btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer;
       font-size: 12px; font-weight: 600; color: #fff; }
.btn:hover { opacity: 0.85; }
.btn-add { background: #2e7d32; width: 100%; margin-bottom: 8px; }
.btn-export { background: #0A1628; width: 100%; padding: 10px; font-size: 14px; }
.btn-sm { padding: 3px 8px; font-size: 11px; background: #888; }
.btn-row { display: flex; gap: 6px; margin-top: 4px; }

.sel-list { max-height: 220px; overflow-y: auto; border: 1px solid #eee;
            border-radius: 4px; font-size: 11px; }
.sel-item { display: flex; justify-content: space-between; padding: 4px 8px;
            border-bottom: 1px solid #f0f0f0; align-items: center; }
.sel-item:hover { background: #f9f9f9; }
.sel-item .x { cursor: pointer; color: #c00; font-weight: bold; padding: 0 4px; }
.empty { color: #999; padding: 8px; text-align: center; font-style: italic; }

.status { font-size: 11px; color: #888; padding: 4px 24px; }
#loading { display: none; position: fixed; top: 0; left: 0; right: 0;
           height: 3px; background: #4caf50; animation: slide 1s infinite; z-index: 999; }
@keyframes slide { 0%{width:0} 50%{width:60%} 100%{width:100%} }
.progress { background: #e0e0e0; border-radius: 4px; height: 18px; overflow: hidden;
            position: relative; }
.progress .bar { background: #1f5fa0; height: 100%; width: 0%;
                 transition: width 0.15s; border-radius: 4px; }
.progress .text { position: absolute; top: 0; left: 0; right: 0; text-align: center;
                  font-size: 11px; line-height: 18px; color: #333; font-weight: 600; }
</style>
</head>
<body>

<div id="loading"></div>

<div class="header">
  <h1>SIGGRAPH Supp — Sequence Curation</h1>
  <p>Curate clips for slides 5 (Kettlebell / Dancing / Walking) and 6
     (image-only vs ours). Per-panel MP4 export, anonymized.</p>
</div>

<div class="controls">
  <div class="ctrl-group">
    <label>Session</label>
    <select id="session"></select>
  </div>
  <div class="ctrl-group nav-row">
    <label>Frame</label>
    <button onclick="prevFrame()">&#9664;</button>
    <span class="frame-num" id="frameNum">0 / 0</span>
    <button onclick="nextFrame()">&#9654;</button>
  </div>
  <div class="ctrl-group">
    <label>Frame</label>
    <input type="range" id="frameSlider" min="0" max="0" value="0" style="width:160px">
  </div>
  <div class="ctrl-group">
    <label>Azimuth</label>
    <div class="range-row">
      <input type="range" id="azim" min="-180" max="180" value="70">
      <span class="val" id="azimVal">70</span>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Elevation</label>
    <div class="range-row">
      <input type="range" id="elev" min="-180" max="180" value="15">
      <span class="val" id="elevVal">15</span>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Line Width</label>
    <div class="range-row">
      <input type="range" id="lineWidth" min="0.5" max="6" value="2.5" step="0.5">
      <span class="val" id="lineWidthVal">2.5</span>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Zoom</label>
    <div class="range-row">
      <input type="range" id="zoom" min="0.3" max="3.0" value="1.0" step="0.1">
      <span class="val" id="zoomVal">1.0</span>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Display</label>
    <div class="chk-row" id="modelChks"><!-- one checkbox per loaded model + Input + GT --></div>
  </div>
  <div class="ctrl-group">
    <div class="chk-row">
      <label><input type="checkbox" id="hideAxes" checked> Hide axes</label>
    </div>
  </div>
</div>

<div class="main">
  <div class="images" id="panels"><!-- panel divs injected dynamically --></div>

  <div class="sidebar">
    <div class="card">
      <h3>MPJPE (mm) — <span id="infoAction">...</span></h3>
      <table class="mpjpe" id="mpjpeTable">
        <tr><th></th><th>Full</th><th>Upper</th><th>Lower</th></tr>
      </table>
    </div>

    <div class="card">
      <h3>Selection (curated clips)</h3>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>Start</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="selStart" value="0" min="0" style="flex:1; padding:4px;">
            <button class="btn btn-sm" onclick="setSel('start')">now</button>
          </div>
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>End</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="selEnd" value="0" min="0" style="flex:1; padding:4px;">
            <button class="btn btn-sm" onclick="setSel('end')">now</button>
          </div>
        </div>
      </div>
      <div class="ctrl-group" style="margin-bottom:6px;">
        <label>Label (optional)</label>
        <input type="text" id="selLabel" placeholder="e.g. Walking, Baseline_Comparison"
               style="padding:4px;">
      </div>
      <div class="chk-row" id="selModelChks" style="margin-bottom:6px;">
        <!-- per-model checkboxes for which models go in this clip -->
      </div>
      <button class="btn btn-add" onclick="selAdd()">+ Add to Selection</button>
      <h3>Curated <span id="selCount">(0)</span></h3>
      <div class="sel-list" id="selList"><div class="empty">Empty</div></div>
      <div class="btn-row">
        <button class="btn btn-sm" onclick="selClear()">Clear</button>
        <button class="btn btn-sm" style="background:#2e7d32"
                onclick="selSave()">Save selections.json</button>
        <button class="btn btn-sm" style="background:#555"
                onclick="selLoad()">Load</button>
      </div>
      <div id="selStatus" style="font-size:11px; color:#666; margin-top:6px;"></div>
    </div>

    <div class="card">
      <h3>Scan sessions (mean MPJPE, toggled models only)</h3>
      <div style="font-size:11px; color:#666; margin-bottom:6px;">
        Per-session mean over every frame, comparing only the model columns
        currently toggled above. Δ = best baseline − ours (when ours is
        toggled), or max−min spread otherwise.
      </div>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>Threshold (mm)</label>
          <input type="number" id="scanThresh" value="3" min="0" step="0.5"
                 style="padding:4px;">
        </div>
        <div class="ctrl-group">
          <label>Scope</label>
          <select id="scanScope" style="padding:4px;">
            <option value="__all__">All sessions</option>
            <option value="__current__">Current session</option>
          </select>
        </div>
      </div>
      <button class="btn" id="scanBtn" style="background:#2e7d32; width:100%; margin-bottom:6px;"
              onclick="doScan()">Scan Frames</button>
      <div class="progress" id="scanProgress" style="display:none; margin-bottom:6px;">
        <div class="bar" id="scanBar"></div>
        <span class="text" id="scanBarText"></span>
      </div>
      <div id="scanStatus" style="font-size:11px; color:#666; margin-bottom:4px;"></div>
      <div class="sel-list" id="scanResults" style="max-height:240px;">
        <div class="empty">Find frames where ours beats image-only.</div>
      </div>
    </div>

    <div class="card">
      <h3>Video Export (MP4, mp4v)</h3>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>Start</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="vidStart" value="0" min="0" style="flex:1; padding:4px;">
            <button class="btn btn-sm" onclick="vidSetCur('start')">now</button>
          </div>
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>End</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="vidEnd" value="0" min="0" style="flex:1; padding:4px;">
            <button class="btn btn-sm" onclick="vidSetCur('end')">now</button>
          </div>
        </div>
      </div>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>FPS</label>
          <input type="number" id="vidFps" value="30" min="1" max="60" step="1"
                 style="width:100%; padding:4px;">
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>Panel (px)</label>
          <input type="number" id="vidPanel" value="1080" min="128" max="2160" step="8"
                 style="width:100%; padding:4px;">
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>DPI</label>
          <input type="number" id="vidDpi" value="200" min="50" max="600" step="50"
                 style="width:100%; padding:4px;">
        </div>
      </div>
      <div class="chk-row" style="margin-bottom:6px;">
        <label><input type="checkbox" id="vidLabels"> Show labels</label>
        <label><input type="checkbox" id="vidAnon" checked> Strip metadata (exiftool)</label>
        <label><input type="checkbox" id="vidGT"> Include GT</label>
      </div>
      <div class="ctrl-group" style="margin-bottom:6px;">
        <label>Filename prefix (optional)</label>
        <input type="text" id="vidName" placeholder="e.g. Walking, baseline" style="padding:4px;">
      </div>
      <div class="chk-row" id="vidModelChks" style="margin-bottom:6px;">
        <!-- per-model render checkboxes -->
      </div>
      <button class="btn" id="vidBtn"
              style="background:#1f5fa0; width:100%; margin-bottom:6px;"
              onclick="doVideo()">Render this range</button>
      <button class="btn" id="vidAllBtn"
              style="background:#0A1628; width:100%; margin-bottom:6px;"
              onclick="doRenderAll()">Render ALL from selections.json</button>
      <div class="progress" id="vidProgress" style="display:none; margin-bottom:6px;">
        <div class="bar" id="vidBar"></div>
        <span class="text" id="vidBarText"></span>
      </div>
      <div id="vidStatus" style="font-size:11px; color:#666; margin-bottom:4px; word-break:break-all;"></div>
      <div id="vidResult"></div>
    </div>
  </div>
</div>

<div class="status" id="status">Ready</div>

<script>
let sessions = [], modelList = [], virtualList = [],
    curSession = '', curFrame = 0,
    maxFrame = 0, renderTimer = null;

async function init() {
  modelList = await (await fetch('/api/models')).json();
  try { virtualList = await (await fetch('/api/virtuals')).json(); }
  catch (e) { virtualList = []; }
  buildModelChecks();
  sessions = await (await fetch('/api/sessions')).json();
  const sel = document.getElementById('session');
  sessions.forEach(s => {
    const o = document.createElement('option');
    o.value = s.name; o.textContent = `${s.action}  (${s.n_frames})`;
    sel.appendChild(o);
  });
  if (sessions.length) {
    curSession = sessions[0].name;
    maxFrame = sessions[0].n_frames - 1;
    document.getElementById('frameSlider').max = maxFrame;
    initRanges();
    render();
  }
}

function buildModelChecks() {
  const colHost = document.getElementById('modelChks');
  const selHost = document.getElementById('selModelChks');
  const vidHost = document.getElementById('vidModelChks');
  let html = '<label><input type="checkbox" data-key="input" class="colChk" checked> Input</label>';
  let selHtml = '<label><input type="checkbox" data-key="input" class="selChk" checked> Input</label>';
  let vidHtml = '<label><input type="checkbox" data-key="input" class="vidChk" checked> Input</label>';
  modelList.forEach(m => {
    const tag = m.color_scheme === 'baseline' ? '🔴' : '🟢';
    const checked = m.key === 'lhf_gbh' ? 'checked' : (m.key === 'image_only' ? 'checked' : '');
    html += `<label><input type="checkbox" data-key="${m.key}" class="colChk" ${checked}> ${tag} ${m.title}</label>`;
    selHtml += `<label><input type="checkbox" data-key="${m.key}" class="selChk" ${m.key==='lhf_gbh'?'checked':''}> ${m.title}</label>`;
    vidHtml += `<label><input type="checkbox" data-key="${m.key}" class="vidChk" ${m.key==='lhf_gbh'?'checked':''}> ${m.title}</label>`;
  });
  // Virtual panels: depth (image), GT pose, GT-HMD trio. Off by default.
  // colChks (preview pane) gets only the visual ones; vid/sel get all of them.
  virtualList.forEach(v => {
    html += `<label><input type="checkbox" data-key="${v.key}" class="colChk"> ${v.title}</label>`;
    selHtml += `<label><input type="checkbox" data-key="${v.key}" class="selChk"> ${v.title}</label>`;
    vidHtml += `<label><input type="checkbox" data-key="${v.key}" class="vidChk"> ${v.title}</label>`;
  });
  html += '<label><input type="checkbox" id="showGT"> GT (gray)</label>';
  colHost.innerHTML = html;
  selHost.innerHTML = selHtml;
  vidHost.innerHTML = vidHtml;
  document.querySelectorAll('.colChk').forEach(el => el.addEventListener('change', render));
  document.getElementById('showGT').addEventListener('change', render);
}

function colKeys() {
  // All non-input checkboxes (models AND virtual panels) — used for the
  // /api/render call. Virtual/model split happens server-side.
  return [...document.querySelectorAll('.colChk')]
    .filter(el => el.checked && el.dataset.key !== 'input')
    .map(el => el.dataset.key);
}
function modelColKeys() {
  // Toggled model columns ONLY — used for scan ranking, which makes no
  // sense for virtual panels (depth/GT/GT-HMD have no MPJPE).
  const virtualSet = new Set(virtualList.map(v => v.key));
  return colKeys().filter(k => !virtualSet.has(k));
}
function showInput() {
  return document.querySelector('.colChk[data-key=input]').checked;
}
function vidKeys() {
  return [...document.querySelectorAll('.vidChk')]
    .filter(el => el.checked).map(el => el.dataset.key);
}
function selKeys() {
  return [...document.querySelectorAll('.selChk')]
    .filter(el => el.checked && el.dataset.key !== 'input')
    .map(el => el.dataset.key);
}

document.getElementById('session').addEventListener('change', e => {
  curSession = e.target.value;
  const s = sessions.find(x => x.name === curSession);
  maxFrame = s ? s.n_frames - 1 : 0;
  curFrame = 0;
  document.getElementById('frameSlider').max = maxFrame;
  document.getElementById('frameSlider').value = 0;
  initRanges();
  render();
});

document.getElementById('frameSlider').addEventListener('input', e => {
  curFrame = parseInt(e.target.value); scheduleRender();
});
document.getElementById('frameSlider').addEventListener('change', () => render());

['azim','elev','lineWidth','zoom'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('input', () => {
    document.getElementById(id+'Val').textContent = el.value; scheduleRender();
  });
  el.addEventListener('change', () => render());
});

document.getElementById('hideAxes').addEventListener('change', () => render());

function prevFrame() { if(curFrame>0){curFrame--;document.getElementById('frameSlider').value=curFrame;render();} }
function nextFrame() { if(curFrame<maxFrame){curFrame++;document.getElementById('frameSlider').value=curFrame;render();} }

document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT'||e.target.tagName==='TEXTAREA') return;
  if(e.key==='ArrowLeft') prevFrame();
  else if(e.key==='ArrowRight') nextFrame();
  else if(e.key==='a') selAdd();
  else if(e.key==='[') setSel('start');
  else if(e.key===']') setSel('end');
});

function scheduleRender() { clearTimeout(renderTimer); renderTimer = setTimeout(render, 220); }

async function render() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('status').textContent = 'Running inference...';
  const params = new URLSearchParams({
    session: curSession, frame: curFrame,
    azim: document.getElementById('azim').value,
    elev: document.getElementById('elev').value,
    show_gt: document.getElementById('showGT').checked,
    show_pred: true,
    line_width: document.getElementById('lineWidth').value,
    zoom: document.getElementById('zoom').value,
    hide_axes: document.getElementById('hideAxes').checked,
    models: colKeys().join(','),
  });
  try {
    const res = await fetch('/api/render?'+params);
    const d = await res.json();
    renderPanels(d);
    document.getElementById('frameNum').textContent = `${curFrame} / ${maxFrame}`;
    document.getElementById('infoAction').textContent =
      `${d.action}  frame ${d.frame_id}`;
    renderMpjpeTable(d.mpjpe);
    document.getElementById('status').textContent = `${d.action}  frame ${d.frame_id}`;
  } catch(err) {
    document.getElementById('status').textContent = 'Error: '+err;
  }
  document.getElementById('loading').style.display = 'none';
}

function renderPanels(d) {
  const host = document.getElementById('panels');
  const enabled = new Set(colKeys());
  let html = '';
  if (showInput() && d.img) {
    html += `<div class="panel input"><img src="data:image/jpeg;base64,${d.img}"><div class="label">Input</div></div>`;
  }
  if (enabled.has('depth') && d.depth) {
    html += `<div class="panel"><img src="data:image/jpeg;base64,${d.depth}"><div class="label">Depth</div></div>`;
  }
  modelList.forEach(m => {
    if (!d.panels[m.key]) return;
    const cls = m.color_scheme === 'baseline' ? 'baseline' : 'ours';
    html += `<div class="panel ${cls}">
               <img src="data:image/png;base64,${d.panels[m.key]}">
               <div class="label">${m.title}</div>
             </div>`;
  });
  virtualList.forEach(v => {
    if (v.kind === 'image') return;     // depth handled above
    if (!d.panels[v.key]) return;
    html += `<div class="panel">
               <img src="data:image/png;base64,${d.panels[v.key]}">
               <div class="label">${v.title}</div>
             </div>`;
  });
  host.innerHTML = html || '<div class="empty">No panels selected</div>';
}

function renderMpjpeTable(mpjpe) {
  let html = '<tr><th></th><th>Full</th><th>Upper</th><th>Lower</th></tr>';
  modelList.forEach(m => {
    const e = mpjpe[m.key];
    if (!e) return;
    const cls = m.color_scheme === 'baseline' ? 'style="color:#c44"' : '';
    html += `<tr ${cls}><td>${m.title}</td><td>${e.full}</td><td>${e.upper}</td><td>${e.lower}</td></tr>`;
  });
  // Δ vs ours
  const ours = mpjpe['lhf_gbh'];
  if (ours) {
    modelList.forEach(m => {
      if (m.key === 'lhf_gbh') return;
      const e = mpjpe[m.key]; if (!e) return;
      const d = (e.full - ours.full).toFixed(1);
      const c = d > 0 ? '#2a7a3a' : (d < 0 ? '#c44' : '#333');
      html += `<tr style="font-weight:bold; color:${c}"><td>Δ ${m.title} − ours</td><td>${d > 0 ? '+' : ''}${d}</td><td>-</td><td>-</td></tr>`;
    });
  }
  document.getElementById('mpjpeTable').innerHTML = html;
}

function initRanges() {
  document.getElementById('selStart').value = 0;
  document.getElementById('selEnd').value = maxFrame;
  document.getElementById('selStart').max = maxFrame;
  document.getElementById('selEnd').max = maxFrame;
  document.getElementById('vidStart').value = 0;
  document.getElementById('vidEnd').value = maxFrame;
  document.getElementById('vidStart').max = maxFrame;
  document.getElementById('vidEnd').max = maxFrame;
}

function setSel(which) {
  document.getElementById(which === 'start' ? 'selStart' : 'selEnd').value = curFrame;
}
function vidSetCur(which) {
  document.getElementById(which === 'start' ? 'vidStart' : 'vidEnd').value = curFrame;
}

async function selAdd() {
  const start = parseInt(document.getElementById('selStart').value);
  const end = parseInt(document.getElementById('selEnd').value);
  const label = document.getElementById('selLabel').value;
  const res = await fetch('/api/sel_add', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session:curSession, start, end, label, models:selKeys()}),
  });
  updateSelList((await res.json()).selected);
}

async function selRemove(idx) {
  const res = await fetch('/api/sel_remove', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index:idx}),
  });
  updateSelList((await res.json()).selected);
}

async function selClear() {
  const res = await fetch('/api/sel_clear', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:'{}'});
  updateSelList((await res.json()).selected);
}

async function selSave() {
  const res = await fetch('/api/sel_save', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:'{}'});
  const d = await res.json();
  document.getElementById('selStatus').textContent =
    `Saved ${d.count} selections to ${d.path}`;
}

async function selLoad() {
  const res = await fetch('/api/sel_load', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:'{}'});
  if (!res.ok) {
    const err = await res.json();
    document.getElementById('selStatus').textContent = `Load failed: ${err.error}`;
    return;
  }
  const d = await res.json();
  updateSelList(d.selected);
  document.getElementById('selStatus').textContent = `Loaded from ${d.path}`;
}

function updateSelList(items) {
  document.getElementById('selCount').textContent = `(${items.length})`;
  const el = document.getElementById('selList');
  if (!items.length) { el.innerHTML = '<div class="empty">Empty</div>'; return; }
  el.innerHTML = items.map((it,i) =>
    `<div class="sel-item">
       <span>${i+1}. <b>${it.action}</b> ${it.start_frame}–${it.end_frame}
       <i style="color:#888">[${it.models.join(',')}]</i></span>
       <span class="x" onclick="selRemove(${i})">&times;</span>
     </div>`).join('');
}

function doScan() {
  const threshold = parseFloat(document.getElementById('scanThresh').value);
  let scope = document.getElementById('scanScope').value;
  if (scope==='__current__') scope = curSession;
  const btn = document.getElementById('scanBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const prog = document.getElementById('scanProgress');
  const bar = document.getElementById('scanBar');
  const barText = document.getElementById('scanBarText');
  prog.style.display = 'block'; bar.style.width = '0%'; barText.textContent = '0%';
  document.getElementById('scanStatus').textContent = '';
  document.getElementById('scanResults').innerHTML = '<div class="empty">Scanning...</div>';

  const toggled = modelColKeys();   // model keys only (virtual panels skipped)
  if (!toggled.length) {
    document.getElementById('scanStatus').textContent =
      'Toggle at least one model column (Image-only / LHF-only / LHF+GBH) before scanning.';
    btn.disabled=false; btn.style.opacity='1';
    prog.style.display='none';
    return;
  }
  const params = new URLSearchParams({threshold, session:scope, models:toggled.join(',')});
  const es = new EventSource('/api/scan?'+params);
  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'progress') {
      const pct = Math.round(d.done/d.total*100);
      bar.style.width = pct+'%';
      barText.textContent = `${d.done}/${d.total} frames (${d.sessions} sessions kept)`;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width = '100%'; barText.textContent = 'Done';
      setTimeout(()=>{prog.style.display='none';}, 800);
      btn.disabled = false; btn.style.opacity = '1';
      const hasOurs = d.models.indexOf('lhf_gbh') !== -1 && d.models.length >= 2;
      const metricLabel = hasOurs ? 'Δ (best baseline − ours)'
        : (d.models.length >= 2 ? 'Δ (max − min mean)' : '−mean (best=largest)');
      document.getElementById('scanStatus').textContent =
        `${d.total} sessions match. Sorted by ${metricLabel}. Models: ${d.models.join(', ')}`;
      const el = document.getElementById('scanResults');
      if (!d.results.length) {
        el.innerHTML = '<div class="empty">No sessions exceed threshold.</div>'; return;
      }
      const titleOf = k => {
        const m = modelList.find(x=>x.key===k); return m ? m.title : k;
      };
      let html = '<table style="width:100%; font-size:11px; border-collapse:collapse;">';
      html += '<thead><tr><th style="text-align:left; padding:3px 6px; background:#f0f0f0;">Session</th><th style="padding:3px 6px; background:#f0f0f0;">N</th>';
      d.models.forEach(k => {
        html += `<th style="padding:3px 6px; background:#f0f0f0;" title="mean Full mm">${titleOf(k)}</th>`;
      });
      html += '<th style="padding:3px 6px; background:#f0f0f0;">Δ Full</th><th style="padding:3px 6px; background:#f0f0f0;">Δ Lower</th></tr></thead><tbody>';
      d.results.forEach(r => {
        html += `<tr style="cursor:pointer;" onclick="jumpToFrame('${r.session}',0)" onmouseover="this.style.background='#f9f9f9'" onmouseout="this.style.background=''">`;
        html += `<td style="padding:3px 6px; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${r.session}"><b>${r.action}</b></td>`;
        html += `<td style="text-align:right; padding:3px 6px;">${r.n_frames}</td>`;
        d.models.forEach(k => {
          const v = r.means[k] ? r.means[k].full : '-';
          html += `<td style="text-align:right; padding:3px 6px;">${v}</td>`;
        });
        const sign = r.metric > 0 ? '+' : '';
        const c = r.metric > 0 ? '#2a7a3a' : (r.metric < 0 ? '#c44' : '#333');
        html += `<td style="text-align:right; padding:3px 6px; color:${c}; font-weight:600;">${sign}${r.metric}</td>`;
        const sign2 = r.metric_lo > 0 ? '+' : '';
        const c2 = r.metric_lo > 0 ? '#2a7a3a' : (r.metric_lo < 0 ? '#c44' : '#333');
        html += `<td style="text-align:right; padding:3px 6px; color:${c2};">${sign2}${r.metric_lo}</td>`;
        html += '</tr>';
      });
      html += '</tbody></table>';
      el.innerHTML = html;
    } else if (d.type === 'error') {
      es.close(); prog.style.display='none';
      btn.disabled=false; btn.style.opacity='1';
      document.getElementById('scanStatus').textContent = 'Error: '+d.msg;
    }
  };
  es.onerror = function() {
    es.close(); prog.style.display='none';
    btn.disabled=false; btn.style.opacity='1';
    document.getElementById('scanStatus').textContent =
      document.getElementById('scanStatus').textContent || 'Scan failed.';
  };
}

function jumpToFrame(session, frameIdx) {
  document.getElementById('session').value = session;
  curSession = session;
  const s = sessions.find(x=>x.name===session);
  maxFrame = s ? s.n_frames-1 : 0;
  curFrame = frameIdx;
  document.getElementById('frameSlider').max = maxFrame;
  document.getElementById('frameSlider').value = curFrame;
  initRanges();
  render();
}

function doVideo() {
  const start = parseInt(document.getElementById('vidStart').value);
  const end = parseInt(document.getElementById('vidEnd').value);
  const fps = parseFloat(document.getElementById('vidFps').value);
  const panel = parseInt(document.getElementById('vidPanel').value);
  const dpi = parseInt(document.getElementById('vidDpi').value);
  const labels = document.getElementById('vidLabels').checked;
  const anon = document.getElementById('vidAnon').checked;
  const gt = document.getElementById('vidGT').checked;
  const name = document.getElementById('vidName').value;
  const status = document.getElementById('vidStatus');
  document.getElementById('vidResult').innerHTML = '';

  if (isNaN(start) || isNaN(end) || end < start) {
    status.textContent = 'Invalid range.'; return;
  }
  if (start < 0 || end > maxFrame) {
    status.textContent = `Out of bounds 0-${maxFrame}.`; return;
  }
  const models = vidKeys();
  if (!models.length) { status.textContent = 'Select at least one model.'; return; }

  const btn = document.getElementById('vidBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const prog = document.getElementById('vidProgress');
  const bar = document.getElementById('vidBar');
  const barText = document.getElementById('vidBarText');
  prog.style.display='block'; bar.style.width='0%';
  barText.textContent = `0 / ${end-start+1}`;
  status.textContent = 'Rendering...';

  const params = new URLSearchParams({
    session: curSession, start, end, fps,
    panel_size: panel, dpi, show_labels: labels,
    show_gt: gt, show_pred: true,
    hide_axes: document.getElementById('hideAxes').checked,
    azim: document.getElementById('azim').value,
    elev: document.getElementById('elev').value,
    line_width: document.getElementById('lineWidth').value,
    zoom: document.getElementById('zoom').value,
    anonymize: anon, filename: name,
    models: models.join(','),
  });
  const es = new EventSource('/api/video_render?'+params);
  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'progress') {
      const pct = Math.round(d.done/d.total*100);
      bar.style.width = pct+'%';
      barText.textContent = `${d.done} / ${d.total}`;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width='100%'; barText.textContent='Done';
      setTimeout(()=>{prog.style.display='none';}, 800);
      btn.disabled=false; btn.style.opacity='1';
      const totalMb = d.files.reduce((a,f)=>a+f.size_mb, 0).toFixed(2);
      status.textContent =
        `Saved ${d.files.length} files (${d.frames} frames @ ${d.fps}fps, ${d.dpi} DPI, ${totalMb} MB total). Anonymized=${d.anonymized}.`;
      document.getElementById('vidResult').innerHTML = d.files.map(f => {
        const url = '/api/video_file/'+encodeURIComponent(f.filename);
        return `<div style="margin-top:8px; padding:6px; background:#fafafa; border:1px solid #eee; border-radius:4px;">
          <div style="font-size:11px; font-weight:600; margin-bottom:4px;">
            ${f.title} <span style="color:#888; font-weight:400;">(${f.size_mb} MB)</span>
          </div>
          <video src="${url}" controls preload="metadata" style="width:100%; background:#000;"></video>
          <div style="margin-top:4px; font-size:11px; word-break:break-all;">
            <a href="${url}" download="${f.filename}" style="color:#1f5fa0;">Download</a>
            — <span style="color:#666;">${f.path}</span>
          </div>
        </div>`;
      }).join('');
    } else if (d.type === 'error') {
      es.close(); prog.style.display='none';
      btn.disabled=false; btn.style.opacity='1';
      status.textContent = 'Error: '+d.msg;
    }
  };
  es.onerror = function() {
    es.close(); prog.style.display='none';
    btn.disabled=false; btn.style.opacity='1';
    if (!status.textContent.startsWith('Error:') && !status.textContent.startsWith('Saved'))
      status.textContent = 'Render connection failed.';
  };
}

function doRenderAll() {
  const fps = parseFloat(document.getElementById('vidFps').value);
  const panel = parseInt(document.getElementById('vidPanel').value);
  const dpi = parseInt(document.getElementById('vidDpi').value);
  const labels = document.getElementById('vidLabels').checked;
  const anon = document.getElementById('vidAnon').checked;
  const status = document.getElementById('vidStatus');
  document.getElementById('vidResult').innerHTML = '';
  const btn = document.getElementById('vidAllBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const prog = document.getElementById('vidProgress');
  const bar = document.getElementById('vidBar');
  const barText = document.getElementById('vidBarText');
  prog.style.display='block'; bar.style.width='0%';
  barText.textContent = '0';
  status.textContent = 'Rendering all selections...';

  const params = new URLSearchParams({
    fps, panel_size: panel, dpi,
    show_labels: labels, anonymize: anon,
    hide_axes: document.getElementById('hideAxes').checked,
    azim: document.getElementById('azim').value,
    elev: document.getElementById('elev').value,
    line_width: document.getElementById('lineWidth').value,
    zoom: document.getElementById('zoom').value,
  });
  let allHtml = '';
  const es = new EventSource('/api/render_selections?'+params);
  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'clip_start') {
      barText.textContent = `clip ${d.index+1}/${d.total}: ${d.prefix} (${d.frames}f)`;
      bar.style.width = `${(d.index/d.total)*100}%`;
    } else if (d.type === 'clip_done') {
      d.files.forEach(f => {
        const url = '/api/video_file/'+encodeURIComponent(f.filename);
        allHtml += `<div style="margin-top:6px; padding:6px; background:#fafafa; border:1px solid #eee; border-radius:4px;">
          <div style="font-size:11px; font-weight:600;">${f.title} — ${f.filename} (${f.size_mb} MB)</div>
          <video src="${url}" controls preload="metadata" style="width:100%; background:#000;"></video>
        </div>`;
      });
      document.getElementById('vidResult').innerHTML = allHtml;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width='100%'; barText.textContent='Done';
      setTimeout(()=>{prog.style.display='none';}, 800);
      btn.disabled=false; btn.style.opacity='1';
      status.textContent = `Saved ${d.count} files total.`;
    } else if (d.type === 'error') {
      status.textContent = 'Error: '+d.msg;
    }
  };
  es.onerror = function() {
    es.close(); prog.style.display='none';
    btn.disabled=false; btn.style.opacity='1';
    if (!status.textContent.startsWith('Saved') && !status.textContent.startsWith('Error:'))
      status.textContent = 'Render-all connection failed.';
  };
}

init();
</script>
</body>
</html>
"""


# ── Entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='SIGGRAPH supplementary video curation GUI',
        epilog=(
            'Per-model path overrides accept either RELATIVE (anchored on '
            f'REPO_ROOT={REPO_ROOT}) or ABSOLUTE paths.\n'
            'Examples:\n'
            '  --model-ckpt lhf_gbh=/mnt/dataset_vol/work_dir_260408/lhf_gbh/best.pth\n'
            '  --model-config lhf_only=my_code/custom_config/HMD_kinect_v5_flag_cascaded_lhf_only_run2_10ep_config.py'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=7862)
    p.add_argument('--no-image-only', action='store_true')
    p.add_argument('--no-lhf-only', action='store_true')
    p.add_argument('--no-lhf-gbh', action='store_true')
    p.add_argument(
        '--model-config', action='append', default=[], metavar='KEY=PATH',
        help='Override a model\'s config (relative or absolute). Repeatable.')
    p.add_argument(
        '--model-ckpt', action='append', default=[], metavar='KEY=PATH',
        help='Override a model\'s checkpoint (relative or absolute). Repeatable.')
    return p.parse_args()


def _apply_path_overrides(spec_list, field):
    """Apply `--model-config` / `--model-ckpt` overrides onto MODELS."""
    for spec in spec_list:
        if '=' not in spec:
            print(f'WARN: ignoring {field} override "{spec}" (expected KEY=PATH)')
            continue
        key, raw = spec.split('=', 1)
        key = key.strip()
        raw = raw.strip()
        if key not in MODELS:
            print(f'WARN: unknown model key "{key}" in --{field.replace("_", "-")} '
                  f'(known: {list(MODELS.keys())})')
            continue
        MODELS[key][field] = raw
        absform = 'abs' if Path(os.path.expanduser(raw)).is_absolute() else 'rel→REPO_ROOT'
        print(f'  override: {key}.{field} = {raw}  [{absform}]')


def main():
    global g_sessions, g_session_frames, g_models, g_pipeline, g_device

    args = parse_args()
    g_device = args.device

    skip = set()
    if args.no_image_only: skip.add('image_only')
    if args.no_lhf_only:   skip.add('lhf_only')
    if args.no_lhf_gbh:    skip.add('lhf_gbh')

    if args.model_config or args.model_ckpt:
        print('Applying path overrides:')
        _apply_path_overrides(args.model_config, 'config')
        _apply_path_overrides(args.model_ckpt, 'ckpt')

    print(f'Loading frames from {args.data_root}...')
    frames = load_all_frames(args.data_root)
    if not frames:
        print('ERROR: No frames found.')
        return

    seen = set()
    for f in frames:
        s = f['session']
        if s not in seen:
            g_sessions.append(s)
            g_session_frames[s] = []
            seen.add(s)
        g_session_frames[s].append(f)
    print(f'  → {len(g_sessions)} sessions, {len(frames)} frames total')

    for k in MODEL_ORDER:
        if k in skip:
            print(f'Skipping {k} (--no-{k.replace("_","-")} flag)')
            continue
        meta = MODELS[k]
        cfg_p = _resolve_path(meta['config'])
        ck_p = _resolve_path(meta['ckpt'])
        if not cfg_p.exists() or not ck_p.exists():
            print(f'WARN: missing files for {k} — skipping')
            print(f'   config={cfg_p}  exists={cfg_p.exists()}')
            print(f'   ckpt={ck_p}    exists={ck_p.exists()}')
            continue
        print(f'Loading {k} ({meta["title"]})')
        print(f'   config={cfg_p}')
        print(f'   ckpt={ck_p}')
        model, _ = load_model(str(cfg_p), str(ck_p), args.device)
        g_models[k] = model

    if not g_models:
        print('ERROR: No models loaded.')
        return

    g_pipeline = build_pipeline()
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f'\n  Models loaded: {list(g_models.keys())}')
    print(f'  Video dir:     {VIDEO_DIR}')
    print(f'  Open http://localhost:{args.port} in your browser.\n')
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
