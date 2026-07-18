"""
Web-based GUI for selecting and generating Figure 7.

Flask app accessible via browser — no display / no Gradio needed.

Features:
  - Browse sessions / frames with dropdown + slider
  - 3D overlay: GT (gray dashed) + Pred (colored) on the same axes
  - Adjustable azimuth / elevation sliders
  - Toggle GT / Pred visibility
  - Add frames to Success / Failure lists, remove items
  - Export publication figure (PNG + PDF)

Usage:
    python my_code/paper_figures/fig7_web_gui.py
    python my_code/paper_figures/fig7_web_gui.py --device cuda:1 --port 7860

Then open http://<server-ip>:7860 in your browser.

Requirements (already in mmpose env):
    flask, pillow, opencv-python, matplotlib, numpy, torch
"""

import argparse
import base64
import io
import json as _json
import os
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_fig7_qualitative import (
    REPO_ROOT, DATA_ROOT,
    CONFIG_LHF, CKPT_LHF, CONFIG_GBH, CKPT_GBH,
    SKELETON, UPPER_INDICES, LOWER_INDICES,
    LINK_COLORS, KPT_COLORS,
    load_all_frames, load_model, build_pipeline,
    run_inference, compute_mpjpe,
)

from flask import Flask, jsonify, request, Response, send_from_directory
from werkzeug.utils import secure_filename

# ── Global state ─────────────────────────────────────────────────────────

g_sessions = []         # ordered session names
g_session_frames = {}   # session_name → [frame, ...]
g_model_lhf = None
g_model_gbh = None
g_pipeline = None
g_device = 'cuda:0'
g_cache = {}            # (session, frame_id) → inference results
g_success = []          # selected success items
g_failure = []          # selected failure items

# ── Video export config ──────────────────────────────────────────────────

VIDEO_DIR = str(REPO_ROOT / 'my_code/my_paper/videos')
g_video_lock = threading.Lock()


# ── Rendering helpers ────────────────────────────────────────────────────

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


def render_3d(gt, pred, elev, azim, show_gt, show_pred, title='',
              line_width=2.5, zoom=1.0, hide_axes=False):
    """Render 3D overlay → base64 PNG string.

    Args:
        zoom: Axis range multiplier. <1 = zoom in, >1 = zoom out.
        hide_axes: If True, make the 3D coordinate frame fully transparent.
    """
    fig = plt.figure(figsize=(4, 3.5), dpi=110)
    ax = fig.add_subplot(111, projection='3d')

    all_pts = []
    if show_gt and gt is not None:
        all_pts.append(gt)
    if show_pred and pred is not None:
        all_pts.append(pred)
    if not all_pts:
        plt.close(fig)
        return ''

    combined = np.concatenate(all_pts, axis=0)
    center = combined.mean(axis=0)
    rng = max(np.abs(combined - center).max(), 0.3) * 1.3 * zoom

    gt_lw = max(line_width * 0.8, 1.0)
    kpt_s_pred = max(16 * line_width, 20)
    kpt_s_gt = max(12 * line_width, 15)

    if show_gt and gt is not None:
        _draw_skeleton(ax, gt,
                       [(0.55, 0.55, 0.55)] * len(SKELETON),
                       [(0.45, 0.45, 0.45)] * 16,
                       lw=gt_lw, kpt_s=kpt_s_gt, alpha=0.5, ls='--',
                       edge_c='gray', edge_w=0.3, zorder=3)

    if show_pred and pred is not None:
        _draw_skeleton(ax, pred,
                       LINK_COLORS, KPT_COLORS,
                       lw=line_width, kpt_s=kpt_s_pred, alpha=1.0, ls='-',
                       edge_c='white', edge_w=0.5, zorder=5)

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


def render_input_image(img_path):
    """Load image → base64 PNG."""
    img = cv2.imread(img_path)
    if img is None:
        return ''
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


# ── Inference ────────────────────────────────────────────────────────────

def get_predictions(session, frame_idx):
    frames = g_session_frames[session]
    frame = frames[frame_idx]
    key = (session, frame['frame_id'])
    if key not in g_cache:
        pred_lhf = run_inference(g_model_lhf, g_pipeline, frame,
                                 frame['hmd_9'], g_device)
        pred_gbh = run_inference(g_model_gbh, g_pipeline, frame,
                                 frame['hmd_12'], g_device)
        fl, ul, ll = compute_mpjpe(pred_lhf, frame['p3d'])
        fg, ug, lg = compute_mpjpe(pred_gbh, frame['p3d'])
        g_cache[key] = dict(pred_lhf=pred_lhf, pred_gbh=pred_gbh,
                            full_lhf=fl, up_lhf=ul, lo_lhf=ll,
                            full_gbh=fg, up_gbh=ug, lo_gbh=lg)
    return g_cache[key]


# ── Export ───────────────────────────────────────────────────────────────

def do_export(elev, azim, line_width=2.5, zoom=1.0, hide_axes=False, dpi=200):
    """Export each selected frame as separate PNG files.

    Saves per-row:
      row{N}_input.png   — egocentric image
      row{N}_lhf.png     — LHF 3D plot (with GT overlay)
      row{N}_gbh.png     — GBH 3D plot (with GT overlay)
      row{N}_gt.png      — GT-only 3D plot
    """
    ns, nf = len(g_success), len(g_failure)
    nr = ns + nf
    if nr == 0:
        return None, None

    out_dir = str(REPO_ROOT / 'my_code/my_paper/revised_paper/figures')
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    all_items = list(g_success) + list(g_failure)

    for ri, item in enumerate(all_items):
        fr = item['frame']
        plhf, pgbh, gt = item['pred_lhf'], item['pred_gbh'], fr['p3d']
        fl, _, ll = compute_mpjpe(plhf, gt)
        fg, _, lg = compute_mpjpe(pgbh, gt)

        tag = 'success' if ri < ns else 'failure'
        prefix = f'row{ri:02d}_{tag}_{fr["action"]}_f{fr["frame_id"]}'

        # Input image
        img_path = os.path.join(out_dir, f'{prefix}_input.png')
        img = cv2.imread(fr['img_path'])
        cv2.imwrite(img_path, img)
        saved_files.append(img_path)

        # LHF 3D plot (with GT overlay)
        lhf_path = os.path.join(out_dir, f'{prefix}_lhf.png')
        _save_3d_plot(lhf_path, gt, plhf, elev, azim,
                      line_width, zoom, hide_axes, dpi)
        saved_files.append(lhf_path)

        # GBH 3D plot (with GT overlay)
        gbh_path = os.path.join(out_dir, f'{prefix}_gbh.png')
        _save_3d_plot(gbh_path, gt, pgbh, elev, azim,
                      line_width, zoom, hide_axes, dpi)
        saved_files.append(gbh_path)

        # GT-only 3D plot
        gt_path = os.path.join(out_dir, f'{prefix}_gt.png')
        _save_3d_plot(gt_path, None, gt, elev, azim,
                      line_width, zoom, hide_axes, dpi)
        saved_files.append(gt_path)

    return out_dir, saved_files


def _save_3d_plot(path, gt, pred, elev, azim,
                  line_width, zoom, hide_axes, dpi=200):
    """Render a single 3D overlay and save to file."""
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    pts_all = []
    if gt is not None:
        pts_all.append(gt)
    if pred is not None:
        pts_all.append(pred)
    if not pts_all:
        plt.close(fig)
        return

    combined = np.concatenate(pts_all)
    center = combined.mean(axis=0)
    rng = max(np.abs(combined - center).max(), 0.3) * 1.3 * zoom

    gt_lw = max(line_width * 0.8, 1.0)
    kpt_s_pred = max(16 * line_width, 20)
    kpt_s_gt = max(12 * line_width, 15)

    if gt is not None:
        _draw_skeleton(ax, gt,
                       [(0.55, 0.55, 0.55)] * len(SKELETON),
                       [(0.45, 0.45, 0.45)] * 16,
                       gt_lw, kpt_s_gt, 0.5, '--', 'gray', 0.3, 3)
    if pred is not None:
        _draw_skeleton(ax, pred, LINK_COLORS, KPT_COLORS,
                       line_width, kpt_s_pred, 1.0, '-', 'white', 0.5, 5)

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

    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05,
                transparent=hide_axes)
    plt.close(fig)


# ── Video rendering ──────────────────────────────────────────────────────

def _render_3d_np(gt, pred, elev, azim, show_gt, show_pred,
                  line_width, zoom, hide_axes, width, height, dpi=100):
    """Render a 3D overlay to a BGR numpy array of exact (height, width).

    Physical figsize is kept constant (5"x5") so linewidths and markers scale
    the same regardless of dpi — dpi then acts as a supersampling/quality
    knob. The rendered image is resized down to (width, height) with area
    interpolation for crisp output.
    """
    figsize_in = 5.0
    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(111, projection='3d')

    all_pts = []
    if show_gt and gt is not None:
        all_pts.append(gt)
    if show_pred and pred is not None:
        all_pts.append(pred)
    if not all_pts:
        plt.close(fig)
        return np.full((height, width, 3), 255, dtype=np.uint8)

    combined = np.concatenate(all_pts, axis=0)
    center = combined.mean(axis=0)
    rng = max(np.abs(combined - center).max(), 0.3) * 1.3 * zoom

    gt_lw = max(line_width * 0.8, 1.0)
    kpt_s_pred = max(16 * line_width, 20)
    kpt_s_gt = max(12 * line_width, 15)

    if show_gt and gt is not None:
        _draw_skeleton(ax, gt,
                       [(0.55, 0.55, 0.55)] * len(SKELETON),
                       [(0.45, 0.45, 0.45)] * 16,
                       gt_lw, kpt_s_gt, 0.5, '--', 'gray', 0.3, 3)
    if show_pred and pred is not None:
        _draw_skeleton(ax, pred, LINK_COLORS, KPT_COLORS,
                       line_width, kpt_s_pred, 1.0, '-', 'white', 0.5, 5)

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


def _make_input_panel(frame, panel_size, show_labels):
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


def _make_3d_panel(gt, pred, elev, azim, show_gt, show_pred,
                   line_width, zoom, hide_axes, panel_size, dpi,
                   title, sub_label, show_labels):
    img = _render_3d_np(gt, pred, elev, azim, show_gt, show_pred,
                        line_width, zoom, hide_axes,
                        panel_size, panel_size, dpi)
    if show_labels:
        img = np.vstack([_label_bar(panel_size, title, sub_label), img])
    return img


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


@app.route('/api/render')
def api_render():
    session = request.args.get('session', g_sessions[0] if g_sessions else '')
    frame_idx = int(request.args.get('frame', 0))
    azim = int(request.args.get('azim', 70))
    elev = int(request.args.get('elev', 15))
    show_gt = request.args.get('show_gt', 'true') == 'true'
    show_pred = request.args.get('show_pred', 'true') == 'true'
    line_width = float(request.args.get('line_width', 2.5))
    zoom = float(request.args.get('zoom', 1.0))
    hide_axes = request.args.get('hide_axes', 'false') == 'true'

    if session not in g_session_frames:
        return jsonify(error='session not found'), 404

    frames = g_session_frames[session]
    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    frame = frames[frame_idx]

    data = get_predictions(session, frame_idx)

    img_b64 = render_input_image(frame['img_path'])

    tl = f"LHF {data['full_lhf']:.0f}mm (L:{data['lo_lhf']:.0f})"
    tg = f"GBH {data['full_gbh']:.0f}mm (L:{data['lo_gbh']:.0f})"

    lhf_b64 = render_3d(
        frame['p3d'] if show_gt else None,
        data['pred_lhf'] if show_pred else None,
        elev, azim, show_gt, show_pred, title=tl,
        line_width=line_width, zoom=zoom, hide_axes=hide_axes)
    gbh_b64 = render_3d(
        frame['p3d'] if show_gt else None,
        data['pred_gbh'] if show_pred else None,
        elev, azim, show_gt, show_pred, title=tg,
        line_width=line_width, zoom=zoom, hide_axes=hide_axes)

    e = data
    return jsonify(
        img=img_b64, lhf=lhf_b64, gbh=gbh_b64,
        action=frame['action'], frame_id=int(frame['frame_id']),
        full_lhf=float(round(e['full_lhf'], 1)),
        up_lhf=float(round(e['up_lhf'], 1)),
        lo_lhf=float(round(e['lo_lhf'], 1)),
        full_gbh=float(round(e['full_gbh'], 1)),
        up_gbh=float(round(e['up_gbh'], 1)),
        lo_gbh=float(round(e['lo_gbh'], 1)),
    )


@app.route('/api/add', methods=['POST'])
def api_add():
    body = request.get_json(force=True)
    session = body['session']
    frame_idx = int(body['frame'])
    target = body['target']  # 'success' or 'failure'

    frames = g_session_frames[session]
    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    frame = frames[frame_idx]
    data = get_predictions(session, frame_idx)

    item = dict(
        frame=frame,
        pred_lhf=data['pred_lhf'].copy(),
        pred_gbh=data['pred_gbh'].copy(),
        errors={k: data[k] for k in
                ('full_lhf', 'up_lhf', 'lo_lhf',
                 'full_gbh', 'up_gbh', 'lo_gbh')},
    )
    (g_success if target == 'success' else g_failure).append(item)
    return jsonify(success=_fmt_list(g_success), failure=_fmt_list(g_failure))


@app.route('/api/remove', methods=['POST'])
def api_remove():
    body = request.get_json(force=True)
    target = body['target']
    idx = int(body['index'])
    lst = g_success if target == 'success' else g_failure
    if 0 <= idx < len(lst):
        lst.pop(idx)
    return jsonify(success=_fmt_list(g_success), failure=_fmt_list(g_failure))


@app.route('/api/clear', methods=['POST'])
def api_clear():
    body = request.get_json(force=True)
    target = body['target']
    (g_success if target == 'success' else g_failure).clear()
    return jsonify(success=_fmt_list(g_success), failure=_fmt_list(g_failure))


@app.route('/api/scan')
def api_scan():
    """Scan frames via Server-Sent Events for live progress.

    Query params: threshold, session (__all__ or name), mode (success|failure|both)
    Streams: progress events, then a final 'done' event with results JSON.
    """
    threshold = float(request.args.get('threshold', 10))
    target_session = request.args.get('session', '__all__')
    mode = request.args.get('mode', 'both')

    scan_sessions = g_sessions if target_session == '__all__' else [target_session]

    # Count total frames to scan
    total = sum(len(g_session_frames.get(s, [])) for s in scan_sessions)

    def generate():
        results = []
        done = 0
        for sess in scan_sessions:
            if sess not in g_session_frames:
                continue
            frames = g_session_frames[sess]
            for fi, frame in enumerate(frames):
                data = get_predictions(sess, fi)
                lo_delta = float(data['lo_lhf'] - data['lo_gbh'])

                if mode == 'success' and lo_delta < threshold:
                    pass
                elif mode == 'failure' and float(data['full_gbh']) < threshold:
                    pass
                elif mode == 'both' and abs(lo_delta) < threshold:
                    pass
                else:
                    results.append(dict(
                        session=sess, frame_idx=fi,
                        frame_id=int(frame['frame_id']),
                        action=frame['action'],
                        full_lhf=float(round(data['full_lhf'], 1)),
                        full_gbh=float(round(data['full_gbh'], 1)),
                        lo_delta=float(round(lo_delta, 1)),
                    ))

                done += 1
                if done % 5 == 0 or done == total:
                    yield f"data: {_json.dumps({'type':'progress','done':done,'total':total,'matches':len(results)})}\n\n"

        if mode == 'success':
            results.sort(key=lambda x: x['lo_delta'], reverse=True)
        elif mode == 'failure':
            results.sort(key=lambda x: x['full_gbh'], reverse=True)
        else:
            results.sort(key=lambda x: abs(x['lo_delta']), reverse=True)

        yield f"data: {_json.dumps({'type':'done','results':results[:100],'total':len(results)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/session_avg')
def api_session_avg():
    """Compute per-session mean MPJPE (LHF vs GBH) across every frame.

    Streams progress then a final 'done' event with one row per session plus
    a weighted 'overall' aggregate across all sessions.
    """
    total = sum(len(g_session_frames.get(s, [])) for s in g_sessions)

    def generate():
        results = []
        done = 0
        for sess in g_sessions:
            frames = g_session_frames.get(sess, [])
            if not frames:
                continue
            sums = {k: 0.0 for k in (
                'full_lhf', 'up_lhf', 'lo_lhf',
                'full_gbh', 'up_gbh', 'lo_gbh')}
            for fi in range(len(frames)):
                data = get_predictions(sess, fi)
                for k in sums:
                    sums[k] += float(data[k])
                done += 1
                if done % 10 == 0 or done == total:
                    yield f"data: {_json.dumps({'type':'progress','done':done,'total':total,'sessions':len(results)})}\n\n"
            n = len(frames)
            avg = {k: v / n for k, v in sums.items()}
            action = '_'.join(sess.split('_')[:-2])
            results.append(dict(
                session=sess, action=action, n_frames=n,
                full_lhf=round(avg['full_lhf'], 1),
                up_lhf=round(avg['up_lhf'], 1),
                lo_lhf=round(avg['lo_lhf'], 1),
                full_gbh=round(avg['full_gbh'], 1),
                up_gbh=round(avg['up_gbh'], 1),
                lo_gbh=round(avg['lo_gbh'], 1),
                full_delta=round(avg['full_lhf'] - avg['full_gbh'], 1),
                up_delta=round(avg['up_lhf'] - avg['up_gbh'], 1),
                lo_delta=round(avg['lo_lhf'] - avg['lo_gbh'], 1),
            ))

        overall = None
        if results:
            tot_n = sum(r['n_frames'] for r in results)
            if tot_n > 0:
                def wavg(key):
                    return sum(r[key] * r['n_frames'] for r in results) / tot_n
                overall = dict(
                    session='__overall__', action='OVERALL', n_frames=tot_n,
                    full_lhf=round(wavg('full_lhf'), 1),
                    up_lhf=round(wavg('up_lhf'), 1),
                    lo_lhf=round(wavg('lo_lhf'), 1),
                    full_gbh=round(wavg('full_gbh'), 1),
                    up_gbh=round(wavg('up_gbh'), 1),
                    lo_gbh=round(wavg('lo_gbh'), 1),
                )
                overall['full_delta'] = round(overall['full_lhf'] - overall['full_gbh'], 1)
                overall['up_delta'] = round(overall['up_lhf'] - overall['up_gbh'], 1)
                overall['lo_delta'] = round(overall['lo_lhf'] - overall['lo_gbh'], 1)

        yield f"data: {_json.dumps({'type':'done','results':results,'overall':overall})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/export', methods=['POST'])
def api_export():
    body = request.get_json(force=True)
    azim = int(body.get('azim', 70))
    elev = int(body.get('elev', 15))
    line_width = float(body.get('line_width', 2.5))
    zoom = float(body.get('zoom', 1.0))
    hide_axes = body.get('hide_axes', False)
    dpi = int(body.get('dpi', 200))
    out_dir, files = do_export(elev, azim, line_width, zoom, hide_axes, dpi)
    if out_dir is None:
        return jsonify(error='Lists are empty'), 400
    return jsonify(out_dir=out_dir, files=files, count=len(files), dpi=dpi)


@app.route('/api/video_render')
def api_video_render():
    """Stream MP4 rendering progress over SSE for [start, end] frame range.

    Produces one MP4 per panel (input / lhf / gbh) reflecting the current
    view settings (azim, elev, line_width, zoom, dpi, show_gt, show_pred,
    hide_axes).
    """
    session = request.args.get('session', g_sessions[0] if g_sessions else '')
    try:
        start = int(request.args.get('start', 0))
        end = int(request.args.get('end', 0))
    except ValueError:
        return Response(
            f"data: {_json.dumps({'type':'error','msg':'Invalid start/end'})}\n\n",
            mimetype='text/event-stream')
    fps = float(request.args.get('fps', 15))
    panel_size = int(request.args.get('panel_size', 480))
    azim = int(request.args.get('azim', 70))
    elev = int(request.args.get('elev', 15))
    show_gt = request.args.get('show_gt', 'true') == 'true'
    show_pred = request.args.get('show_pred', 'true') == 'true'
    line_width = float(request.args.get('line_width', 2.5))
    zoom = float(request.args.get('zoom', 1.0))
    hide_axes = request.args.get('hide_axes', 'false') == 'true'
    show_labels = request.args.get('show_labels', 'true') == 'true'
    dpi = max(50, min(int(request.args.get('dpi', 200)), 600))
    raw_name = (request.args.get('filename', '') or '').strip()

    def err_stream(msg):
        return Response(
            f"data: {_json.dumps({'type':'error','msg':msg})}\n\n",
            mimetype='text/event-stream')

    if session not in g_session_frames:
        return err_stream('Session not found')

    n = len(g_session_frames[session])
    if n == 0:
        return err_stream('Session has no frames')
    start = max(0, min(start, n - 1))
    end = max(start, min(end, n - 1))

    os.makedirs(VIDEO_DIR, exist_ok=True)
    if raw_name:
        prefix = raw_name
        if prefix.lower().endswith('.mp4'):
            prefix = prefix[:-4]
        prefix = secure_filename(prefix)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = secure_filename(f'lhf_gbh_{session}_{start}-{end}_{stamp}')
    if not prefix:
        prefix = f'video_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    panel_keys = ('input', 'lhf', 'gbh')
    panel_titles = {'input': 'Input', 'lhf': 'LHF + GT', 'gbh': 'GBH + GT'}
    paths = {k: os.path.join(VIDEO_DIR, f'{prefix}_{k}.mp4') for k in panel_keys}

    total = end - start + 1

    def build_panels(idx):
        frame = g_session_frames[session][idx]
        data = get_predictions(session, idx)
        sub_lhf = (f"{frame['action']}  f{int(frame['frame_id']):d}  "
                   f"full={data['full_lhf']:.0f}mm  lo={data['lo_lhf']:.0f}mm")
        sub_gbh = (f"{frame['action']}  f{int(frame['frame_id']):d}  "
                   f"full={data['full_gbh']:.0f}mm  lo={data['lo_gbh']:.0f}mm")
        return {
            'input': _make_input_panel(frame, panel_size, show_labels),
            'lhf': _make_3d_panel(frame['p3d'], data['pred_lhf'],
                                  elev, azim, show_gt, show_pred,
                                  line_width, zoom, hide_axes,
                                  panel_size, dpi,
                                  panel_titles['lhf'], sub_lhf, show_labels),
            'gbh': _make_3d_panel(frame['p3d'], data['pred_gbh'],
                                  elev, azim, show_gt, show_pred,
                                  line_width, zoom, hide_axes,
                                  panel_size, dpi,
                                  panel_titles['gbh'], sub_gbh, show_labels),
        }

    def generate():
        with g_video_lock:
            try:
                probe = build_panels(start)
            except Exception as ex:
                yield f"data: {_json.dumps({'type':'error','msg':f'Frame render failed: {ex}'})}\n\n"
                return

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers = {}
            for k in panel_keys:
                h, w = probe[k].shape[:2]
                wr = cv2.VideoWriter(paths[k], fourcc, fps, (w, h))
                if not wr.isOpened():
                    for other in writers.values():
                        other.release()
                    yield f"data: {_json.dumps({'type':'error','msg':f'VideoWriter failed to open for {k} (codec unavailable?)'})}\n\n"
                    return
                writers[k] = wr

            try:
                for k in panel_keys:
                    writers[k].write(probe[k])
                yield f"data: {_json.dumps({'type':'progress','done':1,'total':total})}\n\n"

                for i in range(start + 1, end + 1):
                    try:
                        frames = build_panels(i)
                    except Exception as ex:
                        yield f"data: {_json.dumps({'type':'error','msg':f'Frame {i} failed: {ex}'})}\n\n"
                        return
                    for k in panel_keys:
                        writers[k].write(frames[k])
                    done = i - start + 1
                    if done % 3 == 0 or done == total:
                        yield f"data: {_json.dumps({'type':'progress','done':done,'total':total})}\n\n"
            finally:
                for wr in writers.values():
                    wr.release()

        files = []
        for k in panel_keys:
            p = paths[k]
            try:
                size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)
            except OSError:
                size_mb = 0.0
            files.append(dict(key=k, title=panel_titles[k],
                              filename=os.path.basename(p),
                              path=p, size_mb=size_mb))
        yield f"data: {_json.dumps({'type':'done','files':files,'frames':total,'fps':fps,'dpi':dpi})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/video_file/<path:filename>')
def api_video_file(filename):
    return send_from_directory(VIDEO_DIR, filename, mimetype='video/mp4',
                               as_attachment=False)


def _fmt_list(lst):
    out = []
    for item in lst:
        f = item['frame']
        e = item['errors']
        out.append(dict(
            action=f['action'], frame_id=int(f['frame_id']),
            full_lhf=float(round(e['full_lhf'], 1)),
            full_gbh=float(round(e['full_gbh'], 1)),
            lo_delta=float(round(e['lo_lhf'] - e['lo_gbh'], 1)),
        ))
    return out


# ── Inline HTML/CSS/JS ───────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fig 7 — Qualitative Result Selector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #333; }
.header { background: #1a1a2e; color: #fff; padding: 12px 24px; }
.header h1 { font-size: 18px; font-weight: 600; }
.header p { font-size: 12px; color: #aaa; margin-top: 2px; }

.controls { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 24px;
            background: #fff; border-bottom: 1px solid #ddd; align-items: end; }
.ctrl-group { display: flex; flex-direction: column; gap: 2px; }
.ctrl-group label { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; }
.ctrl-group select, .ctrl-group input[type=range] { min-width: 120px; }
.ctrl-group select { padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; }
.range-row { display: flex; align-items: center; gap: 6px; }
.range-row input[type=range] { width: 140px; }
.range-row .val { font-size: 12px; font-weight: 600; min-width: 32px; text-align: right; }
.chk-row { display: flex; gap: 16px; align-items: center; padding-top: 4px; }
.chk-row label { font-size: 12px; cursor: pointer; }

.nav-row { display: flex; gap: 6px; align-items: end; }
.nav-row button { padding: 5px 14px; border: 1px solid #ccc; border-radius: 4px;
                   background: #fff; cursor: pointer; font-size: 13px; }
.nav-row button:hover { background: #eee; }
.frame-num { font-size: 13px; font-weight: 600; padding: 5px 0; }

.main { display: flex; gap: 16px; padding: 16px 24px; }
.images { flex: 3; display: flex; gap: 8px; }
.images .panel { flex: 1; background: #fff; border-radius: 8px; overflow: hidden;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.images .panel img { width: 100%; display: block; }
.images .panel .label { padding: 6px 10px; font-size: 12px; font-weight: 600;
                         text-align: center; background: #f9f9f9; border-top: 1px solid #eee; }

.sidebar { flex: 1; min-width: 300px; }
.card { background: #fff; border-radius: 8px; padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px; }
.card h3 { font-size: 13px; margin-bottom: 8px; }

table.mpjpe { width: 100%; border-collapse: collapse; font-size: 12px; }
table.mpjpe th, table.mpjpe td { padding: 3px 8px; text-align: right; }
table.mpjpe th { background: #f0f0f0; font-weight: 600; }
table.mpjpe td:first-child, table.mpjpe th:first-child { text-align: left; }

.btn-row { display: flex; gap: 6px; margin-bottom: 8px; }
.btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer;
       font-size: 12px; font-weight: 600; color: #fff; }
.btn-success { background: #2e8b57; }
.btn-failure { background: #cd3333; }
.btn-export { background: #1a1a2e; width: 100%; padding: 10px; font-size: 14px; }
.btn:hover { opacity: 0.85; }
.btn-sm { padding: 3px 8px; font-size: 11px; background: #888; }

.sel-list { max-height: 160px; overflow-y: auto; border: 1px solid #eee;
            border-radius: 4px; font-size: 11px; }
.sel-item { display: flex; justify-content: space-between; padding: 4px 8px;
            border-bottom: 1px solid #f0f0f0; align-items: center; }
.sel-item:hover { background: #f9f9f9; }
.sel-item .x { cursor: pointer; color: #c00; font-weight: bold; padding: 0 4px; }
.empty { color: #999; padding: 8px; text-align: center; font-style: italic; }

.status { font-size: 11px; color: #888; padding: 4px 24px; }
.export-msg { font-size: 12px; color: #2e8b57; margin-top: 8px; }
#loading { display: none; position: fixed; top: 0; left: 0; right: 0;
           height: 3px; background: #3399ff; animation: slide 1s infinite; z-index: 999; }
@keyframes slide { 0%{width:0} 50%{width:60%} 100%{width:100%} }

@media (max-width: 1000px) {
  .main { flex-direction: column; }
  .images { flex-wrap: wrap; }
}
</style>
</head>
<body>

<div id="loading"></div>

<div class="header">
  <h1>Fig 7 — Qualitative Result Selector</h1>
  <p>Browse frames, compare LHF vs GBH overlaid with GT, select for paper figure.</p>
</div>

<div class="controls">
  <div class="ctrl-group">
    <label>Session</label>
    <select id="session"></select>
  </div>
  <div class="ctrl-group nav-row">
    <label>Frame</label>
    <button onclick="prevFrame()">&#9664; Prev</button>
    <span class="frame-num" id="frameNum">0 / 0</span>
    <button onclick="nextFrame()">Next &#9654;</button>
  </div>
  <div class="ctrl-group">
    <label>Frame Slider</label>
    <div class="range-row">
      <input type="range" id="frameSlider" min="0" max="0" value="0">
    </div>
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
    <div class="chk-row">
      <label><input type="checkbox" id="showGT" checked> GT (gray)</label>
      <label><input type="checkbox" id="showPred" checked> Pred (color)</label>
      <label><input type="checkbox" id="hideAxes"> Hide axes</label>
    </div>
    <div class="chk-row">
      <label>Export DPI:</label>
      <select id="exportDpi" style="padding:2px 6px; border:1px solid #ccc; border-radius:4px;">
        <option value="100">100 (draft)</option>
        <option value="150">150</option>
        <option value="200" selected>200</option>
        <option value="300">300 (print)</option>
        <option value="400">400 (high)</option>
      </select>
    </div>
  </div>
</div>

<div class="main">
  <div class="images">
    <div class="panel">
      <img id="imgOut" src="" alt="Input">
      <div class="label">Input Image</div>
    </div>
    <div class="panel">
      <img id="lhfOut" src="" alt="LHF">
      <div class="label" id="lhfLabel">LHF + GT</div>
    </div>
    <div class="panel">
      <img id="gbhOut" src="" alt="GBH">
      <div class="label" id="gbhLabel">GBH + GT</div>
    </div>
  </div>

  <div class="sidebar">
    <div class="card">
      <h3>MPJPE (mm) — <span id="infoAction">...</span></h3>
      <table class="mpjpe">
        <tr><th></th><th>Full</th><th>Upper</th><th>Lower</th></tr>
        <tr><td>LHF</td><td id="eLhfF">-</td><td id="eLhfU">-</td><td id="eLhfL">-</td></tr>
        <tr><td>GBH</td><td id="eGbhF">-</td><td id="eGbhU">-</td><td id="eGbhL">-</td></tr>
        <tr style="font-weight:bold"><td>&Delta;</td><td id="eDF">-</td><td id="eDU">-</td><td id="eDL">-</td></tr>
      </table>
    </div>

    <div class="card">
      <div class="btn-row">
        <button class="btn btn-success" onclick="addTo('success')">+ Success</button>
        <button class="btn btn-failure" onclick="addTo('failure')">+ Failure</button>
      </div>

      <h3>Success Cases <span id="sucCount">(0)</span></h3>
      <div class="sel-list" id="sucList"><div class="empty">Empty</div></div>
      <div class="btn-row" style="margin-top:4px">
        <button class="btn btn-sm" onclick="clearList('success')">Clear</button>
      </div>

      <h3 style="margin-top:10px">Failure Cases <span id="failCount">(0)</span></h3>
      <div class="sel-list" id="failList"><div class="empty">Empty</div></div>
      <div class="btn-row" style="margin-top:4px">
        <button class="btn btn-sm" onclick="clearList('failure')">Clear</button>
      </div>
    </div>

    <div class="card">
      <h3>Scan for L&Delta; Threshold</h3>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:8px;">
        <div class="ctrl-group" style="flex:1">
          <label>Threshold (mm)</label>
          <input type="number" id="scanThresh" value="20" min="0" step="5"
                 style="width:100%; padding:4px; border:1px solid #ccc; border-radius:4px;">
        </div>
        <div class="ctrl-group">
          <label>Mode</label>
          <select id="scanMode" style="padding:4px; border:1px solid #ccc; border-radius:4px;">
            <option value="success">Success (L&Delta; &gt; thresh)</option>
            <option value="failure">Failure (GBH &gt; thresh)</option>
            <option value="both">Both (|L&Delta;| &gt; thresh)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label>Scope</label>
          <select id="scanScope" style="padding:4px; border:1px solid #ccc; border-radius:4px;">
            <option value="__all__">All sessions</option>
            <option value="__current__">Current session</option>
          </select>
        </div>
      </div>
      <button class="btn" id="scanBtn" style="background:#336; width:100%; margin-bottom:6px;" onclick="doScan()">
        Scan Frames
      </button>
      <div id="scanProgress" style="display:none; margin-bottom:6px;">
        <div style="background:#e0e0e0; border-radius:4px; height:18px; overflow:hidden; position:relative;">
          <div id="scanBar" style="background:#336; height:100%; width:0%; transition:width 0.15s; border-radius:4px;"></div>
          <span id="scanBarText" style="position:absolute; top:0; left:0; right:0; text-align:center;
                font-size:11px; line-height:18px; color:#333; font-weight:600;"></span>
        </div>
      </div>
      <div id="scanStatus" style="font-size:11px; color:#666; margin-bottom:4px;"></div>
      <div class="sel-list" id="scanResults" style="max-height:200px;">
        <div class="empty">Click "Scan Frames" to find matching frames.</div>
      </div>
    </div>

    <div class="card">
      <h3>Per-Session Average MPJPE</h3>
      <div style="font-size:11px; color:#666; margin-bottom:6px;">
        Mean MPJPE (LHF vs GBH) per session across every frame. Click a row to jump.
      </div>
      <button class="btn" id="avgBtn"
              style="background:#555; width:100%; margin-bottom:6px;"
              onclick="doSessionAvg()">Compute Session Averages</button>
      <div id="avgProgress" style="display:none; margin-bottom:6px;">
        <div style="background:#e0e0e0; border-radius:4px; height:18px; overflow:hidden; position:relative;">
          <div id="avgBar" style="background:#555; height:100%; width:0%; transition:width 0.15s; border-radius:4px;"></div>
          <span id="avgBarText" style="position:absolute; top:0; left:0; right:0; text-align:center;
                font-size:11px; line-height:18px; color:#333; font-weight:600;"></span>
        </div>
      </div>
      <div id="avgStatus" style="font-size:11px; color:#666; margin-bottom:4px;"></div>
      <div id="avgResults" style="max-height:280px; overflow:auto; border:1px solid #eee; border-radius:4px;">
        <div class="empty">Click "Compute Session Averages" to scan all frames.</div>
      </div>
    </div>

    <div class="card">
      <h3>Video Export (MP4)</h3>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>Start Frame</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="vidStart" value="0" min="0"
                   style="flex:1; padding:4px; border:1px solid #ccc; border-radius:4px;">
            <button class="btn btn-sm" onclick="vidSetCur('start')">now</button>
          </div>
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>End Frame</label>
          <div style="display:flex; gap:4px;">
            <input type="number" id="vidEnd" value="0" min="0"
                   style="flex:1; padding:4px; border:1px solid #ccc; border-radius:4px;">
            <button class="btn btn-sm" onclick="vidSetCur('end')">now</button>
          </div>
        </div>
      </div>
      <div style="display:flex; gap:6px; align-items:end; margin-bottom:6px;">
        <div class="ctrl-group" style="flex:1">
          <label>FPS</label>
          <input type="number" id="vidFps" value="15" min="1" max="60" step="1"
                 style="width:100%; padding:4px; border:1px solid #ccc; border-radius:4px;">
        </div>
        <div class="ctrl-group" style="flex:1">
          <label>Panel (px)</label>
          <input type="number" id="vidPanel" value="480" min="128" max="1024" step="32"
                 style="width:100%; padding:4px; border:1px solid #ccc; border-radius:4px;">
        </div>
        <div class="ctrl-group" style="padding-bottom:4px;">
          <label style="font-size:12px;"><input type="checkbox" id="vidLabels" checked> Labels</label>
        </div>
      </div>
      <div class="ctrl-group" style="margin-bottom:6px;">
        <label>Filename (optional, .mp4)</label>
        <input type="text" id="vidName" placeholder="auto-generated if empty"
               style="width:100%; padding:4px; border:1px solid #ccc; border-radius:4px;">
      </div>
      <button class="btn" id="vidBtn"
              style="background:#663399; width:100%; margin-bottom:6px;"
              onclick="doVideo()">Render Video</button>
      <div id="vidProgress" style="display:none; margin-bottom:6px;">
        <div style="background:#e0e0e0; border-radius:4px; height:18px; overflow:hidden; position:relative;">
          <div id="vidBar" style="background:#663399; height:100%; width:0%; transition:width 0.15s; border-radius:4px;"></div>
          <span id="vidBarText" style="position:absolute; top:0; left:0; right:0; text-align:center;
                font-size:11px; line-height:18px; color:#333; font-weight:600;"></span>
        </div>
      </div>
      <div id="vidStatus" style="font-size:11px; color:#666; margin-bottom:4px; word-break:break-all;"></div>
      <div id="vidResult"></div>
    </div>

    <button class="btn btn-export" onclick="doExport()">Export Figure (PNG + PDF)</button>
    <div class="export-msg" id="exportMsg"></div>
  </div>
</div>

<div class="status" id="status">Ready</div>

<script>
let sessions = [];
let curSession = '';
let curFrame = 0;
let maxFrame = 0;
let renderTimer = null;

async function init() {
  const res = await fetch('/api/sessions');
  sessions = await res.json();
  const sel = document.getElementById('session');
  sessions.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = `${s.action}  (${s.n_frames} frames)`;
    sel.appendChild(opt);
  });
  if (sessions.length > 0) {
    curSession = sessions[0].name;
    maxFrame = sessions[0].n_frames - 1;
    document.getElementById('frameSlider').max = maxFrame;
    vidInitRange();
    render();
  }
}

document.getElementById('session').addEventListener('change', e => {
  curSession = e.target.value;
  const s = sessions.find(x => x.name === curSession);
  maxFrame = s ? s.n_frames - 1 : 0;
  curFrame = 0;
  document.getElementById('frameSlider').max = maxFrame;
  document.getElementById('frameSlider').value = 0;
  vidInitRange();
  render();
});

document.getElementById('frameSlider').addEventListener('input', e => {
  curFrame = parseInt(e.target.value);
  scheduleRender();
});

document.getElementById('frameSlider').addEventListener('change', () => render());

['azim', 'elev', 'lineWidth', 'zoom'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('input', () => {
    document.getElementById(id + 'Val').textContent = el.value;
    scheduleRender();
  });
  el.addEventListener('change', () => render());
});

['showGT', 'showPred', 'hideAxes'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => render());
});

function prevFrame() {
  if (curFrame > 0) { curFrame--; document.getElementById('frameSlider').value = curFrame; render(); }
}
function nextFrame() {
  if (curFrame < maxFrame) { curFrame++; document.getElementById('frameSlider').value = curFrame; render(); }
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') prevFrame();
  else if (e.key === 'ArrowRight') nextFrame();
  else if (e.key === 's') addTo('success');
  else if (e.key === 'f') addTo('failure');
});

function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(render, 200);
}

async function render() {
  const loading = document.getElementById('loading');
  loading.style.display = 'block';
  document.getElementById('status').textContent = 'Running inference...';

  const params = new URLSearchParams({
    session: curSession, frame: curFrame,
    azim: document.getElementById('azim').value,
    elev: document.getElementById('elev').value,
    show_gt: document.getElementById('showGT').checked,
    show_pred: document.getElementById('showPred').checked,
    line_width: document.getElementById('lineWidth').value,
    zoom: document.getElementById('zoom').value,
    hide_axes: document.getElementById('hideAxes').checked,
  });
  try {
    const res = await fetch('/api/render?' + params);
    const d = await res.json();

    document.getElementById('imgOut').src = 'data:image/jpeg;base64,' + d.img;
    document.getElementById('lhfOut').src = 'data:image/png;base64,' + d.lhf;
    document.getElementById('gbhOut').src = 'data:image/png;base64,' + d.gbh;

    document.getElementById('frameNum').textContent = `${curFrame} / ${maxFrame}`;
    document.getElementById('infoAction').textContent = `${d.action}  frame ${d.frame_id}`;

    document.getElementById('eLhfF').textContent = d.full_lhf;
    document.getElementById('eLhfU').textContent = d.up_lhf;
    document.getElementById('eLhfL').textContent = d.lo_lhf;
    document.getElementById('eGbhF').textContent = d.full_gbh;
    document.getElementById('eGbhU').textContent = d.up_gbh;
    document.getElementById('eGbhL').textContent = d.lo_gbh;
    document.getElementById('eDF').textContent = (d.full_lhf - d.full_gbh).toFixed(1);
    document.getElementById('eDU').textContent = (d.up_lhf - d.up_gbh).toFixed(1);
    document.getElementById('eDL').textContent = (d.lo_lhf - d.lo_gbh).toFixed(1);

    document.getElementById('status').textContent = `${d.action}  frame ${d.frame_id}`;
  } catch(err) {
    document.getElementById('status').textContent = 'Error: ' + err;
  }
  loading.style.display = 'none';
}

async function addTo(target) {
  const res = await fetch('/api/add', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session: curSession, frame: curFrame, target })
  });
  const d = await res.json();
  updateLists(d.success, d.failure);
}

async function removeItem(target, idx) {
  const res = await fetch('/api/remove', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ target, index: idx })
  });
  const d = await res.json();
  updateLists(d.success, d.failure);
}

async function clearList(target) {
  const res = await fetch('/api/clear', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ target })
  });
  const d = await res.json();
  updateLists(d.success, d.failure);
}

function updateLists(suc, fail) {
  document.getElementById('sucCount').textContent = `(${suc.length})`;
  document.getElementById('failCount').textContent = `(${fail.length})`;
  renderList('sucList', suc, 'success');
  renderList('failList', fail, 'failure');
}

function renderList(elemId, items, target) {
  const el = document.getElementById(elemId);
  if (items.length === 0) {
    el.innerHTML = '<div class="empty">Empty</div>';
    return;
  }
  el.innerHTML = items.map((it, i) =>
    `<div class="sel-item">
       <span>${i+1}. ${it.action} f${it.frame_id} &nbsp; LHF=${it.full_lhf} GBH=${it.full_gbh} L&Delta;=${it.lo_delta > 0 ? '+' : ''}${it.lo_delta}</span>
       <span class="x" onclick="removeItem('${target}', ${i})">&times;</span>
     </div>`
  ).join('');
}

async function doExport() {
  document.getElementById('exportMsg').textContent = 'Exporting...';
  const res = await fetch('/api/export', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      azim: parseInt(document.getElementById('azim').value),
      elev: parseInt(document.getElementById('elev').value),
      line_width: parseFloat(document.getElementById('lineWidth').value),
      zoom: parseFloat(document.getElementById('zoom').value),
      hide_axes: document.getElementById('hideAxes').checked,
      dpi: parseInt(document.getElementById('exportDpi').value),
    })
  });
  const d = await res.json();
  if (d.error) {
    document.getElementById('exportMsg').textContent = 'Error: ' + d.error;
  } else {
    document.getElementById('exportMsg').textContent =
      `Saved ${d.count} files (${d.dpi} DPI) to ${d.out_dir}`;
  }
}

function vidSetCur(which) {
  document.getElementById(which === 'start' ? 'vidStart' : 'vidEnd').value = curFrame;
}

function vidInitRange() {
  // Default end to maxFrame whenever session changes or on init.
  document.getElementById('vidStart').value = 0;
  document.getElementById('vidEnd').value = maxFrame;
  document.getElementById('vidStart').max = maxFrame;
  document.getElementById('vidEnd').max = maxFrame;
}

function doVideo() {
  const start = parseInt(document.getElementById('vidStart').value);
  const end = parseInt(document.getElementById('vidEnd').value);
  const fps = parseFloat(document.getElementById('vidFps').value);
  const panel = parseInt(document.getElementById('vidPanel').value);
  const labels = document.getElementById('vidLabels').checked;
  const name = document.getElementById('vidName').value;
  const statusEl = document.getElementById('vidStatus');
  const resultEl = document.getElementById('vidResult');
  resultEl.innerHTML = '';

  if (isNaN(start) || isNaN(end) || end < start) {
    statusEl.textContent = 'Invalid range: end must be >= start.';
    return;
  }
  if (start < 0 || end > maxFrame) {
    statusEl.textContent = `Range out of bounds (0-${maxFrame}).`;
    return;
  }
  if (isNaN(fps) || fps <= 0) {
    statusEl.textContent = 'FPS must be positive.';
    return;
  }

  const btn = document.getElementById('vidBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const prog = document.getElementById('vidProgress');
  const bar = document.getElementById('vidBar');
  const barText = document.getElementById('vidBarText');
  prog.style.display = 'block';
  bar.style.width = '0%';
  barText.textContent = `0 / ${end - start + 1}`;
  statusEl.textContent = 'Rendering...';

  const params = new URLSearchParams({
    session: curSession, start, end, fps,
    panel_size: panel,
    show_labels: labels,
    filename: name,
    azim: document.getElementById('azim').value,
    elev: document.getElementById('elev').value,
    show_gt: document.getElementById('showGT').checked,
    show_pred: document.getElementById('showPred').checked,
    line_width: document.getElementById('lineWidth').value,
    zoom: document.getElementById('zoom').value,
    hide_axes: document.getElementById('hideAxes').checked,
    dpi: document.getElementById('exportDpi').value,
  });
  const es = new EventSource('/api/video_render?' + params);
  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'progress') {
      const pct = Math.round(d.done / d.total * 100);
      bar.style.width = pct + '%';
      barText.textContent = `${d.done} / ${d.total} frames`;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width = '100%';
      barText.textContent = 'Done';
      setTimeout(() => { prog.style.display = 'none'; }, 800);
      btn.disabled = false; btn.style.opacity = '1';
      const totalMb = d.files.reduce((a,f) => a + f.size_mb, 0).toFixed(2);
      statusEl.textContent =
        `Saved ${d.files.length} videos (${d.frames} frames @ ${d.fps} fps, ${d.dpi} DPI, ${totalMb} MB total).`;
      resultEl.innerHTML = d.files.map(f => {
        const url = '/api/video_file/' + encodeURIComponent(f.filename);
        return `<div style="margin-top:8px; padding:6px; background:#fafafa; border:1px solid #eee; border-radius:4px;">
          <div style="font-size:11px; font-weight:600; margin-bottom:4px;">
            ${f.title} &nbsp;<span style="color:#888; font-weight:400;">(${f.size_mb} MB)</span>
          </div>
          <video src="${url}" controls preload="metadata"
                 style="width:100%; border-radius:4px; background:#000;"></video>
          <div style="margin-top:4px; font-size:11px; word-break:break-all;">
            <a href="${url}" download="${f.filename}" style="color:#663399; font-weight:600;">Download</a>
            &nbsp;—&nbsp; <span style="color:#666;">${f.path}</span>
          </div>
        </div>`;
      }).join('');
    } else if (d.type === 'error') {
      es.close();
      prog.style.display = 'none';
      btn.disabled = false; btn.style.opacity = '1';
      statusEl.textContent = 'Error: ' + d.msg;
    }
  };
  es.onerror = function() {
    es.close();
    prog.style.display = 'none';
    btn.disabled = false; btn.style.opacity = '1';
    if (!statusEl.textContent.startsWith('Error:') &&
        !statusEl.textContent.startsWith('Saved')) {
      statusEl.textContent = 'Video render connection failed.';
    }
  };
}

function doScan() {
  const threshold = parseFloat(document.getElementById('scanThresh').value);
  const mode = document.getElementById('scanMode').value;
  let scope = document.getElementById('scanScope').value;
  if (scope === '__current__') scope = curSession;

  // Show progress bar, disable button
  const btn = document.getElementById('scanBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const progDiv = document.getElementById('scanProgress');
  const bar = document.getElementById('scanBar');
  const barText = document.getElementById('scanBarText');
  progDiv.style.display = 'block';
  bar.style.width = '0%';
  barText.textContent = '0%';
  document.getElementById('scanStatus').textContent = '';
  document.getElementById('scanResults').innerHTML = '<div class="empty">Scanning...</div>';

  const params = new URLSearchParams({ threshold, mode, session: scope });
  const es = new EventSource('/api/scan?' + params);

  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'progress') {
      const pct = Math.round(d.done / d.total * 100);
      bar.style.width = pct + '%';
      barText.textContent = `${d.done}/${d.total} (${d.matches} matches)`;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width = '100%';
      barText.textContent = 'Done';
      setTimeout(() => { progDiv.style.display = 'none'; }, 800);
      btn.disabled = false; btn.style.opacity = '1';

      document.getElementById('scanStatus').textContent =
        `Found ${d.total} frames (showing top ${Math.min(d.results.length, 100)})`;

      const el = document.getElementById('scanResults');
      if (d.results.length === 0) {
        el.innerHTML = '<div class="empty">No frames match the threshold.</div>';
        return;
      }
      el.innerHTML = d.results.map(r =>
        `<div class="sel-item" style="cursor:pointer"
              onclick="jumpToFrame('${r.session}', ${r.frame_idx})">
           <span>${r.action} f${r.frame_id} &nbsp;
                 LHF=${r.full_lhf} GBH=${r.full_gbh}
                 L&Delta;=${r.lo_delta > 0 ? '+' : ''}${r.lo_delta}</span>
         </div>`
      ).join('');
    }
  };

  es.onerror = function() {
    es.close();
    progDiv.style.display = 'none';
    btn.disabled = false; btn.style.opacity = '1';
    document.getElementById('scanStatus').textContent = 'Scan failed or disconnected.';
  };
}

function doSessionAvg() {
  const btn = document.getElementById('avgBtn');
  btn.disabled = true; btn.style.opacity = '0.5';
  const prog = document.getElementById('avgProgress');
  const bar = document.getElementById('avgBar');
  const barText = document.getElementById('avgBarText');
  prog.style.display = 'block';
  bar.style.width = '0%';
  barText.textContent = '0%';
  document.getElementById('avgStatus').textContent = '';
  document.getElementById('avgResults').innerHTML = '<div class="empty">Computing... first run also populates the prediction cache.</div>';

  const es = new EventSource('/api/session_avg');
  es.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.type === 'progress') {
      const pct = d.total ? Math.round(d.done / d.total * 100) : 0;
      bar.style.width = pct + '%';
      barText.textContent = `${d.done}/${d.total} frames  (${d.sessions} sessions done)`;
    } else if (d.type === 'done') {
      es.close();
      bar.style.width = '100%';
      barText.textContent = 'Done';
      setTimeout(() => { prog.style.display = 'none'; }, 800);
      btn.disabled = false; btn.style.opacity = '1';
      renderAvgTable(d.results, d.overall);
    }
  };
  es.onerror = function() {
    es.close();
    prog.style.display = 'none';
    btn.disabled = false; btn.style.opacity = '1';
    document.getElementById('avgStatus').textContent = 'Scan failed or disconnected.';
  };
}

function renderAvgTable(results, overall) {
  const el = document.getElementById('avgResults');
  if (!results || !results.length) {
    el.innerHTML = '<div class="empty">No sessions.</div>';
    return;
  }
  const th = t => `<th style="padding:4px 6px; background:#f0f0f0; position:sticky; top:0; font-size:10px; white-space:nowrap;">${t}</th>`;
  const deltaCell = v => {
    const sign = v > 0 ? '+' : '';
    const color = v > 0 ? '#2a7a3a' : (v < 0 ? '#c44' : '#333');
    return `<td style="padding:3px 6px; text-align:right; color:${color}; font-weight:600;">${sign}${v}</td>`;
  };
  const num = v => `<td style="padding:3px 6px; text-align:right;">${v}</td>`;
  const row = r => `
    <tr onclick="jumpToFrame('${r.session}', 0)" style="cursor:pointer;" onmouseover="this.style.background='#f9f9f9'" onmouseout="this.style.background=''">
      <td style="padding:3px 6px; text-align:left; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${r.session}">${r.action}</td>
      ${num(r.n_frames)}${num(r.full_lhf)}${num(r.lo_lhf)}${num(r.full_gbh)}${num(r.lo_gbh)}${deltaCell(r.full_delta)}${deltaCell(r.lo_delta)}
    </tr>`;
  const overallRow = overall ? `
    <tr style="font-weight:700; background:#eef;">
      <td style="padding:3px 6px; text-align:left;">${overall.action}</td>
      ${num(overall.n_frames)}${num(overall.full_lhf)}${num(overall.lo_lhf)}${num(overall.full_gbh)}${num(overall.lo_gbh)}${deltaCell(overall.full_delta)}${deltaCell(overall.lo_delta)}
    </tr>` : '';
  el.innerHTML = `
    <table style="width:100%; border-collapse:collapse; font-size:11px;">
      <thead><tr>${th('Session')}${th('N')}${th('LHF-F')}${th('LHF-L')}${th('GBH-F')}${th('GBH-L')}${th('ΔF')}${th('ΔL')}</tr></thead>
      <tbody>${results.map(row).join('')}${overallRow}</tbody>
    </table>`;
  const total = overall ? overall.n_frames : results.reduce((a, r) => a + r.n_frames, 0);
  document.getElementById('avgStatus').textContent =
    `${results.length} sessions, ${total} frames.  ΔF,ΔL = LHF − GBH (positive means GBH is better).`;
}

function jumpToFrame(session, frameIdx) {
  // Switch session dropdown
  const sel = document.getElementById('session');
  sel.value = session;
  curSession = session;
  const s = sessions.find(x => x.name === session);
  maxFrame = s ? s.n_frames - 1 : 0;
  curFrame = frameIdx;
  document.getElementById('frameSlider').max = maxFrame;
  document.getElementById('frameSlider').value = curFrame;
  vidInitRange();
  render();
}

init();
</script>
</body>
</html>
"""

# ── Entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Fig 7 web GUI (Flask)')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=7860)
    return p.parse_args()


def main():
    global g_sessions, g_session_frames
    global g_model_lhf, g_model_gbh, g_pipeline, g_device

    args = parse_args()
    g_device = args.device

    print('Loading dataset frames...')
    frames = load_all_frames(args.data_root)
    if not frames:
        print('ERROR: No frames found. Check --data-root.')
        return

    seen = set()
    for f in frames:
        s = f['session']
        if s not in seen:
            g_sessions.append(s)
            g_session_frames[s] = []
            seen.add(s)
        g_session_frames[s].append(f)

    print('Loading LHF model...')
    g_model_lhf, _ = load_model(CONFIG_LHF, CKPT_LHF, args.device)
    print('Loading GBH model...')
    g_model_gbh, _ = load_model(CONFIG_GBH, CKPT_GBH, args.device)
    g_pipeline = build_pipeline()

    print(f'\n  Open http://localhost:{args.port} in your browser.\n')
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
