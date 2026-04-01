"""
Generate Figure 7: Qualitative results on our real-world dataset.

Layout: 4 columns x N rows
  Column 1: Egocentric image (input)
  Column 2: LHF only (9-dim) 3D pose prediction
  Column 3: LHF + GBH (12-dim) 3D pose prediction
  Column 4: Ground Truth 3D pose

Top rows  : Success cases (GBH significantly improves lower body)
Bottom rows: Failure cases (fast motions / severe occlusion)

Usage:
    python my_code/paper_figures/generate_fig7_qualitative.py

    # Custom selection
    python my_code/paper_figures/generate_fig7_qualitative.py \
        --success-rows 3 --failure-rows 2 --output output_fig7/fig7.pdf
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmpose.registry import MODELS

# ── Constants ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = '/mnt/express1m2/egodataset_for_paper/0223hyeonghwan_batch_ego'

CONFIG_LHF = str(REPO_ROOT / 'my_code/custom_config/HMD_kinect_v5_flag_cascaded_baseline_10ep_config.py')
CKPT_LHF = str(REPO_ROOT / 'work_dirs/HMD_kinect_v5_flag_cascaded_baseline_10ep/best_xregopose_Full Body_All_mpjpe_epoch_9.pth')

CONFIG_GBH = str(REPO_ROOT / 'my_code/custom_config/HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py')
CKPT_GBH = str(REPO_ROOT / 'work_dirs/HMD_kinect_v5_flag_cascaded_ground_info_10ep/best_xregopose_Full Body_All_mpjpe_epoch_10.pth')

# Kinect 32 → xRegopose 16 joint mapping
KINECT_TO_XREGOPOSE = [
    'SPINE_CHEST', 'HEAD',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',
    'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
    'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT',
    'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT',
]
WRIST_FALLBACK = {'WRIST_LEFT': 'HAND_LEFT', 'WRIST_RIGHT': 'HAND_RIGHT'}

SKELETON = [
    (0, 1), (0, 2), (2, 3), (3, 4),         # spine→head, left arm
    (0, 5), (5, 6), (6, 7),                   # right arm
    (0, 8), (8, 9), (9, 10), (10, 11),        # left leg
    (0, 12), (12, 13), (13, 14), (14, 15),    # right leg
]

# Upper body: 0-7, Lower body: 8-15
UPPER_INDICES = list(range(8))
LOWER_INDICES = list(range(8, 16))

# Colors (RGB 0-1) for skeleton links
LINK_COLORS = [
    (0.2, 0.6, 1.0),  # spine-head
    (0.2, 0.6, 1.0),  # spine-Larm
    (0.0, 0.8, 0.3),  # L forearm
    (0.0, 0.8, 0.3),  # L hand
    (0.2, 0.6, 1.0),  # spine-Rarm
    (1.0, 0.5, 0.0),  # R forearm
    (1.0, 0.5, 0.0),  # R hand
    (0.2, 0.6, 1.0),  # spine-Lhip
    (0.0, 0.8, 0.3),  # L knee
    (0.0, 0.8, 0.3),  # L ankle
    (0.0, 0.8, 0.3),  # L foot
    (0.2, 0.6, 1.0),  # spine-Rhip
    (1.0, 0.5, 0.0),  # R knee
    (1.0, 0.5, 0.0),  # R ankle
    (1.0, 0.5, 0.0),  # R foot
]

KPT_COLORS = [
    (0.2, 0.6, 1.0),  # Spine2
    (0.2, 0.6, 1.0),  # Head
    (0.2, 0.6, 1.0),  # L shoulder
    (0.0, 0.8, 0.3),  # L elbow
    (0.0, 0.8, 0.3),  # L hand
    (0.2, 0.6, 1.0),  # R shoulder
    (1.0, 0.5, 0.0),  # R elbow
    (1.0, 0.5, 0.0),  # R hand
    (0.2, 0.6, 1.0),  # L hip
    (0.0, 0.8, 0.3),  # L knee
    (0.0, 0.8, 0.3),  # L ankle
    (0.0, 0.8, 0.3),  # L foot
    (0.2, 0.6, 1.0),  # R hip
    (1.0, 0.5, 0.0),  # R knee
    (1.0, 0.5, 0.0),  # R ankle
    (1.0, 0.5, 0.0),  # R foot
]


# ── Data loading helpers ─────────────────────────────────────────────────

def load_csv(csv_path):
    """Load synced_data.csv → {frame_id: row_dict}."""
    lookup = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[int(row['frame'])] = row
    return lookup


def preprocess_hmd_9dim(head, right_hand, left_hand):
    """Compute 9-dim LHF from head/hand positions (meters)."""
    midpoint = (right_hand + left_hand) / 2.0
    z_axis = midpoint - head
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm if z_norm > 1e-6 else np.array([0., 0., 1.])

    hand_vector = right_hand - left_hand
    x_axis = np.cross(z_axis, hand_vector)
    x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm if x_norm > 1e-6 else np.array([1., 0., 0.])

    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack((x_axis, y_axis, z_axis))

    right_local = rot.T @ (right_hand - head)
    left_local = rot.T @ (left_hand - head)

    return np.concatenate([
        right_local, left_local,
        [np.linalg.norm(right_local - left_local),
         np.linalg.norm(right_local),
         np.linalg.norm(left_local)]
    ]).astype(np.float32)


def parse_frame(json_path, csv_row, img_path):
    """Parse a single frame → (p3d, hmd_9, hmd_12, p2d, vis)."""
    with open(json_path) as f:
        data = json.load(f)
    if data.get('num_bodies', 0) == 0:
        return None

    skel_2d = {j['name']: j for j in data['skeleton_2d']}
    skel_3d = {j['name']: j for j in data['skeleton_3d']}

    p2d = np.zeros((16, 2), dtype=np.float32)
    p3d = np.zeros((16, 3), dtype=np.float32)
    vis = np.zeros(16, dtype=np.float32)

    for xr_idx, kname in enumerate(KINECT_TO_XREGOPOSE):
        j2d = skel_2d.get(kname)
        j3d = skel_3d.get(kname)
        if j2d is None or j3d is None:
            continue
        conf = j3d.get('confidence', 0)
        if kname in WRIST_FALLBACK and conf < 1:
            fb = WRIST_FALLBACK[kname]
            if fb in skel_2d and fb in skel_3d:
                j2d, j3d = skel_2d[fb], skel_3d[fb]
                conf = j3d.get('confidence', 0)
        p2d[xr_idx] = [j2d['u'], j2d['v']]
        p3d[xr_idx] = [j3d['x'], j3d['y'], j3d['z']]
        vis[xr_idx] = 1.0 if conf >= 2 else 0.0

    p3d /= 1000.0  # mm → meters

    head = np.array([float(csv_row['hmd_pos_x']),
                     float(csv_row['hmd_pos_y']),
                     float(csv_row['hmd_pos_z'])], dtype=np.float32)
    left = np.array([float(csv_row['left_pos_x']),
                     float(csv_row['left_pos_y']),
                     float(csv_row['left_pos_z'])], dtype=np.float32)
    right = np.array([float(csv_row['right_pos_x']),
                      float(csv_row['right_pos_y']),
                      float(csv_row['right_pos_z'])], dtype=np.float32)

    hmd_9 = preprocess_hmd_9dim(head, right, left)
    hmd_12 = np.concatenate([
        hmd_9,
        [float(csv_row['hmd_pos_y']),
         float(csv_row['left_pos_y']),
         float(csv_row['right_pos_y'])]
    ]).astype(np.float32)

    return dict(p3d=p3d, p2d=p2d, vis=vis,
                hmd_9=hmd_9, hmd_12=hmd_12, img_path=img_path)


def load_all_frames(data_root):
    """Load all frames from all sessions under data_root.
    Returns list of dicts with keys: p3d, hmd_9, hmd_12, img_path, action, session, frame_id.
    """
    frames = []
    for entry in sorted(os.listdir(data_root)):
        session_dir = os.path.join(data_root, entry)
        if not os.path.isdir(session_dir):
            continue
        csv_path = os.path.join(session_dir, 'synced_data.csv')
        ann_dir = os.path.join(session_dir, 'ego_dataset', 'annotations')
        img_dir = os.path.join(session_dir, 'ego_dataset', 'images')
        if not os.path.isfile(csv_path) or not os.path.isdir(ann_dir):
            continue

        csv_data = load_csv(csv_path)
        action = '_'.join(entry.split('_')[:-2])

        for ann_name in sorted(os.listdir(ann_dir)):
            if not ann_name.endswith('.json'):
                continue
            frame_id = int(ann_name.replace('frame_', '').replace('.json', ''))
            if frame_id not in csv_data:
                continue

            json_path = os.path.join(ann_dir, ann_name)
            img_path = os.path.join(img_dir, ann_name.replace('.json', '.jpg'))
            if not os.path.isfile(img_path):
                continue

            result = parse_frame(json_path, csv_data[frame_id], img_path)
            if result is None:
                continue
            result['action'] = action
            result['session'] = entry
            result['frame_id'] = frame_id
            frames.append(result)

    print(f'Loaded {len(frames)} frames from {data_root}')
    return frames


# ── Model loading and inference ──────────────────────────────────────────

def load_model(config_path, checkpoint_path, device='cuda:0'):
    """Build model from config and load checkpoint."""
    cfg = Config.fromfile(config_path)
    init_default_scope('mmpose')

    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Dataset meta
    from mmpose.datasets.datasets.body3d.egopose_info import dataset_info
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    metainfo = dict(from_file='mmpose/datasets/datasets/body3d/egopose_info.py')
    model.dataset_meta = parse_pose_metainfo(metainfo)
    return model, cfg


def build_pipeline():
    """Build inference pipeline (same for both models)."""
    pipeline_cfg = [
        dict(type='LoadImage'),
        dict(type='EgoImageResize', input_size=(256, 256)),
        dict(
            encoder=dict(
                heatmap_size=(47, 47), input_size=(256, 256),
                sigma=3, type='Custom_mo2cap2_MSRAHeatmap'),
            type='GenerateTarget'),
        dict(type='PackPoseInputs',
             meta_keys=('id', 'img_id', 'img_path', 'category_id',
                        'crowd_index', 'ori_shape', 'img_shape',
                        'input_size', 'input_center', 'input_scale',
                        'flip', 'flip_direction', 'flip_indices',
                        'raw_ann_info', 'dataset_name', 'action')),
    ]
    return Compose(pipeline_cfg)


def run_inference(model, pipeline, frame, hmd_info, device='cuda:0'):
    """Run inference on a single frame. Returns predicted 3D pose (16, 3)."""
    img_path = frame['img_path']
    p2d = frame['p2d']
    p3d = frame['p3d']
    vis = frame['vis']

    data_info = {
        'img_path': img_path,
        'bbox': np.array([[0, 0, 1920, 1080]], dtype=np.float32),
        'bbox_score': np.ones(1, dtype=np.float32),
        'keypoints': p2d.reshape(1, 16, 2),
        'keypoint3d': p3d.reshape(1, 16, 3),
        'keypoints_visible': vis.reshape(1, 16),
        'action': frame['action'],
    }
    data_info.update(model.dataset_meta)

    data = pipeline(data_info)

    # Inject HMD info
    hmd_tensor = torch.from_numpy(hmd_info.reshape(1, -1).astype(np.float32))
    data['data_samples'].gt_instance_labels.set_field(hmd_tensor, 'hmd_info')
    # Need GT for the loss-based decode path
    data['data_samples'].gt_instance_labels.set_field(
        torch.from_numpy(p3d.reshape(1, 16, 3).astype(np.float32)), 'keypoint3d')

    batch = pseudo_collate([data])
    with torch.no_grad():
        results = model.test_step(batch)

    pred_3d = results[0].pred_instances.keypoint_3d.cpu().numpy()
    if pred_3d.ndim == 3:
        pred_3d = pred_3d[0]  # (16, 3)
    return pred_3d


def compute_mpjpe(pred, gt):
    """Per-joint MPJPE in mm. Returns (full, upper, lower)."""
    err = np.linalg.norm(pred - gt, axis=-1) * 1000  # meters → mm
    return err.mean(), err[UPPER_INDICES].mean(), err[LOWER_INDICES].mean()


# ── Visualization ────────────────────────────────────────────────────────

def draw_3d_pose(ax, pose, title='', color_mode='pred', alpha=1.0):
    """Draw 3D skeleton on a matplotlib 3D axis.

    Args:
        ax: Axes3D
        pose: (16, 3) array
        title: subplot title
        color_mode: 'pred' for colored, 'gt' for gray
        alpha: transparency
    """
    if color_mode == 'gt':
        kpt_c = [(0.4, 0.4, 0.4)] * 16
        link_c = [(0.5, 0.5, 0.5)] * len(SKELETON)
    else:
        kpt_c = KPT_COLORS
        link_c = LINK_COLORS

    for idx, (i, j) in enumerate(SKELETON):
        pts = pose[[i, j]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=link_c[idx], linewidth=2.5, alpha=alpha)

    for i in range(16):
        ax.scatter(pose[i, 0], pose[i, 1], pose[i, 2],
                   c=[kpt_c[i]], s=40, alpha=alpha, edgecolors='white',
                   linewidths=0.5, zorder=5)

    center = pose.mean(axis=0)
    max_range = max(np.abs(pose - center).max(), 0.3) * 1.3
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.view_init(elev=15, azim=70)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    if title:
        ax.set_title(title, fontsize=9, pad=-2)


def generate_figure(success_frames, failure_frames, output_path):
    """Generate the publication figure.

    Args:
        success_frames: list of (frame, pred_lhf, pred_gbh) tuples
        failure_frames: list of (frame, pred_lhf, pred_gbh) tuples
        output_path: path to save figure
    """
    n_success = len(success_frames)
    n_failure = len(failure_frames)
    n_rows = n_success + n_failure
    n_cols = 4  # Image, LHF, GBH, GT

    fig = plt.figure(figsize=(10, 2.8 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  wspace=0.02, hspace=0.15,
                  left=0.02, right=0.98, top=0.95, bottom=0.02)

    # Column headers
    col_titles = ['Input Image', 'LHF (9-dim)', 'LHF + GBH (12-dim)', 'Ground Truth']

    all_frames = list(success_frames) + list(failure_frames)

    for row_idx, (frame, pred_lhf, pred_gbh) in enumerate(all_frames):
        gt_3d = frame['p3d']
        action = frame['action']
        fid = frame['frame_id']

        # Compute errors for annotation
        _, _, lhf_lower = compute_mpjpe(pred_lhf, gt_3d)
        _, _, gbh_lower = compute_mpjpe(pred_gbh, gt_3d)
        full_lhf, _, _ = compute_mpjpe(pred_lhf, gt_3d)
        full_gbh, _, _ = compute_mpjpe(pred_gbh, gt_3d)

        # Col 0: Input image
        ax_img = fig.add_subplot(gs[row_idx, 0])
        img = cv2.imread(frame['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.axis('off')
        label = action.replace('-', '\n')
        ax_img.text(0.02, 0.98, label, transform=ax_img.transAxes,
                    fontsize=7, color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
        if row_idx == 0:
            ax_img.set_title(col_titles[0], fontsize=10, fontweight='bold')

        # Col 1: LHF prediction
        ax_lhf = fig.add_subplot(gs[row_idx, 1], projection='3d')
        draw_3d_pose(ax_lhf, pred_lhf)
        err_text = f'{full_lhf:.0f}mm (L:{lhf_lower:.0f})'
        ax_lhf.text2D(0.5, -0.02, err_text, transform=ax_lhf.transAxes,
                       fontsize=7, ha='center', color='#555555')
        if row_idx == 0:
            ax_lhf.set_title(col_titles[1], fontsize=10, fontweight='bold')

        # Col 2: GBH prediction
        ax_gbh = fig.add_subplot(gs[row_idx, 2], projection='3d')
        draw_3d_pose(ax_gbh, pred_gbh)
        err_text = f'{full_gbh:.0f}mm (L:{gbh_lower:.0f})'
        ax_gbh.text2D(0.5, -0.02, err_text, transform=ax_gbh.transAxes,
                       fontsize=7, ha='center', color='#555555')
        if row_idx == 0:
            ax_gbh.set_title(col_titles[2], fontsize=10, fontweight='bold')

        # Col 3: Ground truth
        ax_gt = fig.add_subplot(gs[row_idx, 3], projection='3d')
        draw_3d_pose(ax_gt, gt_3d, color_mode='gt')
        if row_idx == 0:
            ax_gt.set_title(col_titles[3], fontsize=10, fontweight='bold')

    # Add row group labels
    if n_success > 0 and n_failure > 0:
        # "Success" label
        y_success = 1.0 - (n_success * 0.5) / n_rows
        fig.text(0.005, y_success, 'Success\ncases',
                 fontsize=9, fontweight='bold', color='#2E8B57',
                 va='center', rotation=90)
        # "Failure" label
        y_failure = 1.0 - (n_success + n_failure * 0.5) / n_rows
        fig.text(0.005, y_failure, 'Failure\ncases',
                 fontsize=9, fontweight='bold', color='#CD3333',
                 va='center', rotation=90)
        # Divider line
        y_div = 1.0 - n_success / n_rows
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [y_div, y_div],
            transform=fig.transFigure,
            color='gray', linewidth=0.8, linestyle='--'))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {output_path}')

    # Also save as PDF for LaTeX
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'Saved PDF to {pdf_path}')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Fig 7')
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='my_code/my_paper/revised_paper/figures/fig7_qualitative.png')
    parser.add_argument('--success-rows', type=int, default=3,
                        help='Number of success case rows')
    parser.add_argument('--failure-rows', type=int, default=2,
                        help='Number of failure case rows')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = args.device

    # 1. Load all frames
    print('Loading dataset frames...')
    frames = load_all_frames(args.data_root)

    # 2. Load both models
    print('Loading LHF model...')
    model_lhf, _ = load_model(CONFIG_LHF, CKPT_LHF, device)
    print('Loading GBH model...')
    model_gbh, _ = load_model(CONFIG_GBH, CKPT_GBH, device)
    pipeline = build_pipeline()

    # 3. Run inference on all frames, compute per-frame errors
    print(f'Running inference on {len(frames)} frames...')
    scored_frames = []
    for i, frame in enumerate(frames):
        if i % 50 == 0:
            print(f'  {i}/{len(frames)}')

        pred_lhf = run_inference(model_lhf, pipeline, frame, frame['hmd_9'], device)
        pred_gbh = run_inference(model_gbh, pipeline, frame, frame['hmd_12'], device)

        full_lhf, upper_lhf, lower_lhf = compute_mpjpe(pred_lhf, frame['p3d'])
        full_gbh, upper_gbh, lower_gbh = compute_mpjpe(pred_gbh, frame['p3d'])

        scored_frames.append(dict(
            frame=frame,
            pred_lhf=pred_lhf,
            pred_gbh=pred_gbh,
            full_lhf=full_lhf,
            full_gbh=full_gbh,
            lower_lhf=lower_lhf,
            lower_gbh=lower_gbh,
            lower_improvement=lower_lhf - lower_gbh,  # positive = GBH better
        ))

    # 4. Select success cases: largest lower body improvement by GBH
    #    AND GBH full body error is reasonable (< 80mm)
    success_pool = [s for s in scored_frames
                    if s['lower_improvement'] > 0 and s['full_gbh'] < 80]
    success_pool.sort(key=lambda s: s['lower_improvement'], reverse=True)

    # Pick from diverse actions
    selected_success = []
    seen_actions = set()
    for s in success_pool:
        action = s['frame']['action']
        if action not in seen_actions:
            selected_success.append(s)
            seen_actions.add(action)
        if len(selected_success) >= args.success_rows:
            break
    # Fill remaining from top if not enough diverse actions
    if len(selected_success) < args.success_rows:
        for s in success_pool:
            if s not in selected_success:
                selected_success.append(s)
            if len(selected_success) >= args.success_rows:
                break

    # 5. Select failure cases: both models have high error (fast motion / occlusion)
    failure_pool = [s for s in scored_frames if s['full_gbh'] > 80]
    failure_pool.sort(key=lambda s: s['full_gbh'], reverse=True)

    selected_failure = []
    seen_actions_f = set()
    for s in failure_pool:
        action = s['frame']['action']
        if action not in seen_actions_f:
            selected_failure.append(s)
            seen_actions_f.add(action)
        if len(selected_failure) >= args.failure_rows:
            break
    if len(selected_failure) < args.failure_rows:
        for s in failure_pool:
            if s not in selected_failure:
                selected_failure.append(s)
            if len(selected_failure) >= args.failure_rows:
                break

    # 6. Print selection summary
    print('\n=== Selected Success Cases ===')
    for s in selected_success:
        f = s['frame']
        print(f"  {f['action']} frame {f['frame_id']}: "
              f"LHF={s['full_lhf']:.1f}mm (L:{s['lower_lhf']:.1f}) → "
              f"GBH={s['full_gbh']:.1f}mm (L:{s['lower_gbh']:.1f})  "
              f"[lower imp: {s['lower_improvement']:.1f}mm]")

    print('\n=== Selected Failure Cases ===')
    for s in selected_failure:
        f = s['frame']
        print(f"  {f['action']} frame {f['frame_id']}: "
              f"LHF={s['full_lhf']:.1f}mm  GBH={s['full_gbh']:.1f}mm")

    # 7. Generate figure
    success_tuples = [(s['frame'], s['pred_lhf'], s['pred_gbh'])
                      for s in selected_success]
    failure_tuples = [(s['frame'], s['pred_lhf'], s['pred_gbh'])
                      for s in selected_failure]

    generate_figure(success_tuples, failure_tuples, args.output)

    print('\nDone!')


if __name__ == '__main__':
    main()
