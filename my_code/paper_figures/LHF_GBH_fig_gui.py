"""
Interactive GUI for selecting and generating Figure 7.

Features:
  - Browse all sessions / frames with prev/next navigation
  - 3D overlay view: GT (gray) and Pred (colored) on the same axes
  - Adjustable azimuth / elevation sliders for the 3D view
  - Add frames to Success or Failure lists
  - Reorder / remove selected frames
  - Export the final publication figure (PNG + PDF)

Usage:
    python my_code/paper_figures/fig7_gui.py
    python my_code/paper_figures/fig7_gui.py --device cuda:1
    python my_code/paper_figures/fig7_gui.py --data-root /path/to/data
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


# ── 3D drawing with overlay support ──────────────────────────────────────

def draw_pose_overlay(ax, gt, pred, elev, azim, title='',
                      show_gt=True, show_pred=True):
    """Draw GT (gray) and pred (colored) on the same 3D axis."""
    ax.cla()

    # Determine axis limits from both poses
    all_pts = []
    if show_gt and gt is not None:
        all_pts.append(gt)
    if show_pred and pred is not None:
        all_pts.append(pred)
    if not all_pts:
        return
    combined = np.concatenate(all_pts, axis=0)
    center = combined.mean(axis=0)
    max_range = max(np.abs(combined - center).max(), 0.3) * 1.3

    # Draw GT first (behind)
    if show_gt and gt is not None:
        gt_link_c = [(0.55, 0.55, 0.55)] * len(SKELETON)
        gt_kpt_c = [(0.45, 0.45, 0.45)] * 16
        for idx, (i, j) in enumerate(SKELETON):
            pts = gt[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=gt_link_c[idx], linewidth=2.0, alpha=0.5,
                    linestyle='--')
        for i in range(16):
            ax.scatter(gt[i, 0], gt[i, 1], gt[i, 2],
                       c=[gt_kpt_c[i]], s=30, alpha=0.5,
                       edgecolors='gray', linewidths=0.3, zorder=3)

    # Draw Pred on top
    if show_pred and pred is not None:
        for idx, (i, j) in enumerate(SKELETON):
            pts = pred[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=LINK_COLORS[idx], linewidth=2.5, alpha=1.0)
        for i in range(16):
            ax.scatter(pred[i, 0], pred[i, 1], pred[i, 2],
                       c=[KPT_COLORS[i]], s=40, alpha=1.0,
                       edgecolors='white', linewidths=0.5, zorder=5)

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    if title:
        ax.set_title(title, fontsize=9, pad=2)


# ── Export figure (same layout as generate_fig7) ─────────────────────────

def export_figure(success_list, failure_list, output_path, elev, azim):
    """Generate publication figure from selected frames."""
    n_success = len(success_list)
    n_failure = len(failure_list)
    n_rows = n_success + n_failure
    if n_rows == 0:
        return

    n_cols = 4
    fig = plt.figure(figsize=(10, 2.8 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  wspace=0.02, hspace=0.15,
                  left=0.02, right=0.98, top=0.95, bottom=0.02)

    col_titles = ['Input Image', 'LHF (9-dim)', 'LHF + GBH (12-dim)',
                  'Ground Truth']
    all_items = list(success_list) + list(failure_list)

    for row_idx, item in enumerate(all_items):
        frame = item['frame']
        pred_lhf = item['pred_lhf']
        pred_gbh = item['pred_gbh']
        gt_3d = frame['p3d']

        full_lhf, _, lower_lhf = compute_mpjpe(pred_lhf, gt_3d)
        full_gbh, _, lower_gbh = compute_mpjpe(pred_gbh, gt_3d)

        # Col 0 - image
        ax_img = fig.add_subplot(gs[row_idx, 0])
        img = cv2.cvtColor(cv2.imread(frame['img_path']), cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.axis('off')
        label = frame['action'].replace('-', '\n')
        ax_img.text(0.02, 0.98, label, transform=ax_img.transAxes,
                    fontsize=7, color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
        if row_idx == 0:
            ax_img.set_title(col_titles[0], fontsize=10, fontweight='bold')

        # Col 1 - LHF with GT overlay
        ax_lhf = fig.add_subplot(gs[row_idx, 1], projection='3d')
        draw_pose_overlay(ax_lhf, gt_3d, pred_lhf, elev, azim)
        ax_lhf.text2D(0.5, -0.02,
                       f'{full_lhf:.0f}mm (L:{lower_lhf:.0f})',
                       transform=ax_lhf.transAxes, fontsize=7,
                       ha='center', color='#555555')
        if row_idx == 0:
            ax_lhf.set_title(col_titles[1], fontsize=10, fontweight='bold')

        # Col 2 - GBH with GT overlay
        ax_gbh = fig.add_subplot(gs[row_idx, 2], projection='3d')
        draw_pose_overlay(ax_gbh, gt_3d, pred_gbh, elev, azim)
        ax_gbh.text2D(0.5, -0.02,
                       f'{full_gbh:.0f}mm (L:{lower_gbh:.0f})',
                       transform=ax_gbh.transAxes, fontsize=7,
                       ha='center', color='#555555')
        if row_idx == 0:
            ax_gbh.set_title(col_titles[2], fontsize=10, fontweight='bold')

        # Col 3 - GT only
        ax_gt = fig.add_subplot(gs[row_idx, 3], projection='3d')
        draw_pose_overlay(ax_gt, None, gt_3d, elev, azim,
                          show_gt=False, show_pred=True)
        if row_idx == 0:
            ax_gt.set_title(col_titles[3], fontsize=10, fontweight='bold')

    # Row group labels
    if n_success > 0 and n_failure > 0:
        y_s = 1.0 - (n_success * 0.5) / n_rows
        fig.text(0.005, y_s, 'Success\ncases', fontsize=9,
                 fontweight='bold', color='#2E8B57', va='center',
                 rotation=90)
        y_f = 1.0 - (n_success + n_failure * 0.5) / n_rows
        fig.text(0.005, y_f, 'Failure\ncases', fontsize=9,
                 fontweight='bold', color='#CD3333', va='center',
                 rotation=90)
        y_div = 1.0 - n_success / n_rows
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [y_div, y_div],
            transform=fig.transFigure,
            color='gray', linewidth=0.8, linestyle='--'))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return output_path, pdf_path


# ── Main GUI ─────────────────────────────────────────────────────────────

class FigureBuilderApp:

    def __init__(self, root, frames, model_lhf, model_gbh, pipeline, device):
        self.root = root
        self.root.title('Fig 7 — Qualitative Result Selector')
        self.root.geometry('1500x900')

        self.all_frames = frames
        self.model_lhf = model_lhf
        self.model_gbh = model_gbh
        self.pipeline = pipeline
        self.device = device

        # Group frames by session
        self.sessions = []
        self.session_frames = {}
        seen = set()
        for f in frames:
            s = f['session']
            if s not in seen:
                self.sessions.append(s)
                self.session_frames[s] = []
                seen.add(s)
            self.session_frames[s].append(f)

        # State
        self.cur_session_idx = 0
        self.cur_frame_idx = 0
        self.azim = tk.IntVar(value=70)
        self.elev = tk.IntVar(value=15)
        self.show_gt = tk.BooleanVar(value=True)
        self.show_pred = tk.BooleanVar(value=True)

        # Current inference cache
        self.cached_pred_lhf = None
        self.cached_pred_gbh = None
        self.cached_errors = {}

        # Selected frames for export
        self.success_list = []  # list of dicts
        self.failure_list = []

        self._build_ui()
        self._load_current_frame()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        # Top bar: session selector + frame nav
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Label(top, text='Session:').pack(side=tk.LEFT)
        self.session_combo = ttk.Combobox(
            top, values=self._session_labels(), width=50, state='readonly')
        self.session_combo.current(0)
        self.session_combo.pack(side=tk.LEFT, padx=4)
        self.session_combo.bind('<<ComboboxSelected>>', self._on_session_change)

        ttk.Label(top, text='  Frame:').pack(side=tk.LEFT)
        self.frame_spin = ttk.Spinbox(
            top, from_=0, to=0, width=6,
            command=self._on_frame_spin)
        self.frame_spin.pack(side=tk.LEFT, padx=4)

        self.frame_label = ttk.Label(top, text='/ 0')
        self.frame_label.pack(side=tk.LEFT)

        ttk.Button(top, text='< Prev', command=self._prev_frame).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(top, text='Next >', command=self._next_frame).pack(
            side=tk.LEFT, padx=4)

        self.status_label = ttk.Label(top, text='', foreground='gray')
        self.status_label.pack(side=tk.RIGHT, padx=8)

        # Main area: left = matplotlib canvas, right = controls + selection
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        # Matplotlib canvas (left)
        canvas_frame = ttk.Frame(main)
        main.add(canvas_frame, weight=3)

        self.fig = plt.Figure(figsize=(11, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel
        right = ttk.Frame(main, width=380)
        main.add(right, weight=1)

        # --- View controls ---
        ctrl = ttk.LabelFrame(right, text='3D View Controls')
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(ctrl, text='Azimuth').grid(row=0, column=0, sticky='w',
                                             padx=4, pady=2)
        azim_scale = ttk.Scale(ctrl, from_=-180, to=180,
                               variable=self.azim, orient=tk.HORIZONTAL,
                               command=lambda _: self._redraw())
        azim_scale.grid(row=0, column=1, sticky='ew', padx=4)
        self.azim_val = ttk.Label(ctrl, text='70')
        self.azim_val.grid(row=0, column=2, padx=4)

        ttk.Label(ctrl, text='Elevation').grid(row=1, column=0, sticky='w',
                                               padx=4, pady=2)
        elev_scale = ttk.Scale(ctrl, from_=-90, to=90,
                               variable=self.elev, orient=tk.HORIZONTAL,
                               command=lambda _: self._redraw())
        elev_scale.grid(row=1, column=1, sticky='ew', padx=4)
        self.elev_val = ttk.Label(ctrl, text='15')
        self.elev_val.grid(row=1, column=2, padx=4)

        ctrl.columnconfigure(1, weight=1)

        chk_frame = ttk.Frame(ctrl)
        chk_frame.grid(row=2, column=0, columnspan=3, pady=4)
        ttk.Checkbutton(chk_frame, text='Show GT (gray dashed)',
                        variable=self.show_gt,
                        command=self._redraw).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(chk_frame, text='Show Pred (colored)',
                        variable=self.show_pred,
                        command=self._redraw).pack(side=tk.LEFT, padx=8)

        # --- Error info ---
        self.info_frame = ttk.LabelFrame(right, text='MPJPE (mm)')
        self.info_frame.pack(fill=tk.X, padx=4, pady=4)
        self.info_text = tk.Text(self.info_frame, height=5, width=40,
                                 font=('Consolas', 9), state='disabled',
                                 bg='#f8f8f8')
        self.info_text.pack(fill=tk.X, padx=4, pady=4)

        # --- Add buttons ---
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_frame, text='+ Add to Success',
                   command=self._add_success).pack(side=tk.LEFT, padx=4,
                                                   expand=True, fill=tk.X)
        ttk.Button(btn_frame, text='+ Add to Failure',
                   command=self._add_failure).pack(side=tk.LEFT, padx=4,
                                                   expand=True, fill=tk.X)

        # --- Selection lists ---
        sel_nb = ttk.Notebook(right)
        sel_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Success tab
        suc_frame = ttk.Frame(sel_nb)
        sel_nb.add(suc_frame, text='Success Cases')
        self.suc_listbox = tk.Listbox(suc_frame, font=('Consolas', 9))
        self.suc_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        suc_sb = ttk.Scrollbar(suc_frame, orient=tk.VERTICAL,
                               command=self.suc_listbox.yview)
        suc_sb.pack(fill=tk.Y, side=tk.RIGHT)
        self.suc_listbox.config(yscrollcommand=suc_sb.set)

        suc_btn = ttk.Frame(right)
        suc_btn.pack(fill=tk.X, padx=4)
        ttk.Button(suc_btn, text='Remove Selected',
                   command=lambda: self._remove_selected('success')).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(suc_btn, text='Move Up',
                   command=lambda: self._move_item('success', -1)).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(suc_btn, text='Move Down',
                   command=lambda: self._move_item('success', 1)).pack(
            side=tk.LEFT, padx=2)

        # Failure tab
        fail_frame = ttk.Frame(sel_nb)
        sel_nb.add(fail_frame, text='Failure Cases')
        self.fail_listbox = tk.Listbox(fail_frame, font=('Consolas', 9))
        self.fail_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        fail_sb = ttk.Scrollbar(fail_frame, orient=tk.VERTICAL,
                                command=self.fail_listbox.yview)
        fail_sb.pack(fill=tk.Y, side=tk.RIGHT)
        self.fail_listbox.config(yscrollcommand=fail_sb.set)

        fail_btn = ttk.Frame(right)
        fail_btn.pack(fill=tk.X, padx=4)
        ttk.Button(fail_btn, text='Remove Selected',
                   command=lambda: self._remove_selected('failure')).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(fail_btn, text='Move Up',
                   command=lambda: self._move_item('failure', -1)).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(fail_btn, text='Move Down',
                   command=lambda: self._move_item('failure', 1)).pack(
            side=tk.LEFT, padx=2)

        # Export button
        ttk.Separator(right).pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(right, text='Export Figure',
                   command=self._export).pack(padx=4, pady=4, fill=tk.X)

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<s>', lambda e: self._add_success())
        self.root.bind('<f>', lambda e: self._add_failure())

    # ── Helpers ──────────────────────────────────────────────────────

    def _session_labels(self):
        labels = []
        for s in self.sessions:
            action = '_'.join(s.split('_')[:-2])
            n = len(self.session_frames[s])
            labels.append(f'{action}  ({n} frames)  [{s}]')
        return labels

    def _cur_session_name(self):
        return self.sessions[self.cur_session_idx]

    def _cur_frames(self):
        return self.session_frames[self._cur_session_name()]

    def _cur_frame(self):
        return self._cur_frames()[self.cur_frame_idx]

    # ── Navigation callbacks ─────────────────────────────────────────

    def _on_session_change(self, event=None):
        self.cur_session_idx = self.session_combo.current()
        self.cur_frame_idx = 0
        n = len(self._cur_frames())
        self.frame_spin.config(to=max(n - 1, 0))
        self.frame_spin.set(0)
        self.frame_label.config(text=f'/ {n - 1}')
        self._load_current_frame()

    def _on_frame_spin(self):
        try:
            val = int(self.frame_spin.get())
        except ValueError:
            return
        n = len(self._cur_frames())
        val = max(0, min(val, n - 1))
        self.cur_frame_idx = val
        self._load_current_frame()

    def _prev_frame(self):
        if self.cur_frame_idx > 0:
            self.cur_frame_idx -= 1
            self.frame_spin.set(self.cur_frame_idx)
            self._load_current_frame()

    def _next_frame(self):
        if self.cur_frame_idx < len(self._cur_frames()) - 1:
            self.cur_frame_idx += 1
            self.frame_spin.set(self.cur_frame_idx)
            self._load_current_frame()

    # ── Inference + display ──────────────────────────────────────────

    def _load_current_frame(self):
        frame = self._cur_frame()
        self.status_label.config(text='Running inference...')
        self.root.update_idletasks()

        # Run inference
        pred_lhf = run_inference(
            self.model_lhf, self.pipeline, frame, frame['hmd_9'], self.device)
        pred_gbh = run_inference(
            self.model_gbh, self.pipeline, frame, frame['hmd_12'], self.device)

        self.cached_pred_lhf = pred_lhf
        self.cached_pred_gbh = pred_gbh

        full_lhf, up_lhf, lo_lhf = compute_mpjpe(pred_lhf, frame['p3d'])
        full_gbh, up_gbh, lo_gbh = compute_mpjpe(pred_gbh, frame['p3d'])
        self.cached_errors = dict(
            full_lhf=full_lhf, up_lhf=up_lhf, lo_lhf=lo_lhf,
            full_gbh=full_gbh, up_gbh=up_gbh, lo_gbh=lo_gbh)

        self._update_info()
        self._redraw()
        self.status_label.config(
            text=f'{frame["action"]}  frame {frame["frame_id"]}')

    def _update_info(self):
        e = self.cached_errors
        txt = (
            f'         Full   Upper  Lower\n'
            f'LHF   {e["full_lhf"]:6.1f}  {e["up_lhf"]:6.1f}  {e["lo_lhf"]:6.1f}\n'
            f'GBH   {e["full_gbh"]:6.1f}  {e["up_gbh"]:6.1f}  {e["lo_gbh"]:6.1f}\n'
            f'Delta {e["full_lhf"]-e["full_gbh"]:+6.1f}  '
            f'{e["up_lhf"]-e["up_gbh"]:+6.1f}  '
            f'{e["lo_lhf"]-e["lo_gbh"]:+6.1f}'
        )
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', txt)
        self.info_text.config(state='disabled')

    def _redraw(self):
        """Redraw the matplotlib canvas with current frame data."""
        frame = self._cur_frame()
        gt = frame['p3d']
        pred_lhf = self.cached_pred_lhf
        pred_gbh = self.cached_pred_gbh
        if pred_lhf is None or pred_gbh is None:
            return

        elev = self.elev.get()
        azim = self.azim.get()
        show_gt = self.show_gt.get()
        show_pred = self.show_pred.get()

        self.azim_val.config(text=str(azim))
        self.elev_val.config(text=str(elev))

        self.fig.clf()

        # Layout: 1 row, 3 cols  [Image | LHF+GT | GBH+GT]
        gs = GridSpec(1, 3, figure=self.fig, wspace=0.05,
                      left=0.01, right=0.99, top=0.92, bottom=0.02)

        # Col 0 - egocentric image
        ax_img = self.fig.add_subplot(gs[0, 0])
        img = cv2.cvtColor(cv2.imread(frame['img_path']), cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Input Image', fontsize=10)

        # Col 1 - LHF + GT overlay
        ax_lhf = self.fig.add_subplot(gs[0, 1], projection='3d')
        e = self.cached_errors
        draw_pose_overlay(ax_lhf,
                          gt if show_gt else None,
                          pred_lhf if show_pred else None,
                          elev, azim,
                          title=f'LHF  {e["full_lhf"]:.0f}mm '
                                f'(L:{e["lo_lhf"]:.0f})')

        # Col 2 - GBH + GT overlay
        ax_gbh = self.fig.add_subplot(gs[0, 2], projection='3d')
        draw_pose_overlay(ax_gbh,
                          gt if show_gt else None,
                          pred_gbh if show_pred else None,
                          elev, azim,
                          title=f'GBH  {e["full_gbh"]:.0f}mm '
                                f'(L:{e["lo_gbh"]:.0f})')

        self.canvas.draw_idle()

    # ── Selection list management ────────────────────────────────────

    def _make_item(self):
        frame = self._cur_frame()
        return dict(
            frame=frame,
            pred_lhf=self.cached_pred_lhf.copy(),
            pred_gbh=self.cached_pred_gbh.copy(),
            errors=dict(self.cached_errors),
        )

    def _item_label(self, item):
        f = item['frame']
        e = item['errors']
        return (f'{f["action"]:20s} f{f["frame_id"]:03d}  '
                f'LHF={e["full_lhf"]:.0f}  GBH={e["full_gbh"]:.0f}  '
                f'Ldelta={e["lo_lhf"]-e["lo_gbh"]:+.0f}')

    def _add_success(self):
        item = self._make_item()
        self.success_list.append(item)
        self.suc_listbox.insert(tk.END, self._item_label(item))

    def _add_failure(self):
        item = self._make_item()
        self.failure_list.append(item)
        self.fail_listbox.insert(tk.END, self._item_label(item))

    def _remove_selected(self, which):
        if which == 'success':
            sel = self.suc_listbox.curselection()
            if sel:
                idx = sel[0]
                self.success_list.pop(idx)
                self.suc_listbox.delete(idx)
        else:
            sel = self.fail_listbox.curselection()
            if sel:
                idx = sel[0]
                self.failure_list.pop(idx)
                self.fail_listbox.delete(idx)

    def _move_item(self, which, direction):
        lst = self.success_list if which == 'success' else self.failure_list
        lb = self.suc_listbox if which == 'success' else self.fail_listbox
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(lst):
            return
        lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
        # Refresh listbox
        lb.delete(0, tk.END)
        for item in lst:
            lb.insert(tk.END, self._item_label(item))
        lb.selection_set(new_idx)

    # ── Export ───────────────────────────────────────────────────────

    def _export(self):
        if not self.success_list and not self.failure_list:
            messagebox.showwarning('Empty', 'Add frames to Success or '
                                   'Failure lists first.')
            return

        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf')],
            initialfile='fig7_qualitative.png',
            initialdir=str(REPO_ROOT / 'my_code/my_paper/revised_paper/figures'))
        if not path:
            return

        self.status_label.config(text='Exporting figure...')
        self.root.update_idletasks()

        png, pdf = export_figure(
            self.success_list, self.failure_list, path,
            self.elev.get(), self.azim.get())

        self.status_label.config(text=f'Saved: {os.path.basename(png)}')
        messagebox.showinfo('Exported',
                            f'PNG: {png}\nPDF: {pdf}')


# ── Entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Fig 7 interactive GUI')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()

    print('Loading dataset frames...')
    frames = load_all_frames(args.data_root)
    if not frames:
        print('ERROR: No frames found. Check --data-root.')
        return

    print('Loading LHF model...')
    model_lhf, _ = load_model(CONFIG_LHF, CKPT_LHF, args.device)
    print('Loading GBH model...')
    model_gbh, _ = load_model(CONFIG_GBH, CKPT_GBH, args.device)
    pipeline = build_pipeline()

    print('Starting GUI...')
    root = tk.Tk()
    app = FigureBuilderApp(
        root, frames, model_lhf, model_gbh, pipeline, args.device)
    root.mainloop()


if __name__ == '__main__':
    main()
