"""
Visualize the Kinect EgoPose dataset used for training.

1) 3D Joint Mapping: shows how the 32 Azure Kinect joints map to the
   16 xRegopose joints used by custom_kinect_egopose_dataset.py.
2) HMD Data: shows how the HMD (head + two controllers) positions from
   synced_data.csv are incorporated and transformed into the 9-dim HMD info.

Outputs saved to: visualize_kinect_output/
"""

import os, json, csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ─── Paths ────────────────────────────────────────────────────────────
DATA_ROOT = '/mnt/dataset_vol/annotation_egodataset_2026_sei/Train'
OUT_DIR = '/home/hyeonghwan/github/mmpose/visualize_kinect_output'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Kinect → xRegopose mapping (from custom_kinect_egopose_dataset.py) ──
KINECT_TO_XREGOPOSE = [
    'SPINE_CHEST',       # 0: Spine2
    'HEAD',              # 1: Head
    'SHOULDER_LEFT',     # 2: LeftArm
    'ELBOW_LEFT',        # 3: LeftForeArm
    'WRIST_LEFT',        # 4: LeftHand
    'SHOULDER_RIGHT',    # 5: RightArm
    'ELBOW_RIGHT',       # 6: RightForeArm
    'WRIST_RIGHT',       # 7: RightHand
    'HIP_LEFT',          # 8: LeftUpLeg
    'KNEE_LEFT',         # 9: LeftLeg
    'ANKLE_LEFT',        # 10: LeftFoot
    'FOOT_LEFT',         # 11: LeftToeBase
    'HIP_RIGHT',         # 12: RightUpLeg
    'KNEE_RIGHT',        # 13: RightLeg
    'ANKLE_RIGHT',       # 14: RightFoot
    'FOOT_RIGHT',        # 15: RightToeBase
]

XREGOPOSE_NAMES = [
    'Spine2', 'Head',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
]

# Skeleton links (from egopose_info.py) as index pairs
SKELETON_LINKS = [
    (0, 1),   # Spine2 → Head
    (0, 2),   # Spine2 → LeftArm
    (2, 3),   # LeftArm → LeftForeArm
    (3, 4),   # LeftForeArm → LeftHand
    (0, 5),   # Spine2 → RightArm
    (5, 6),   # RightArm → RightForeArm
    (6, 7),   # RightForeArm → RightHand
    (0, 8),   # Spine2 → LeftUpLeg
    (8, 9),   # LeftUpLeg → LeftLeg
    (9, 10),  # LeftLeg → LeftFoot
    (10, 11), # LeftFoot → LeftToeBase
    (0, 12),  # Spine2 → RightUpLeg
    (12, 13), # RightUpLeg → RightLeg
    (13, 14), # RightLeg → RightFoot
    (14, 15), # RightFoot → RightToeBase
]

# Links between ALL Kinect 32 joints (for full-skeleton display)
KINECT_32_LINKS = [
    ('PELVIS', 'SPINE_NAVEL'),
    ('SPINE_NAVEL', 'SPINE_CHEST'),
    ('SPINE_CHEST', 'NECK'),
    ('NECK', 'HEAD'),
    ('NECK', 'CLAVICLE_LEFT'),
    ('CLAVICLE_LEFT', 'SHOULDER_LEFT'),
    ('SHOULDER_LEFT', 'ELBOW_LEFT'),
    ('ELBOW_LEFT', 'WRIST_LEFT'),
    ('WRIST_LEFT', 'HAND_LEFT'),
    ('HAND_LEFT', 'HANDTIP_LEFT'),
    ('HAND_LEFT', 'THUMB_LEFT'),
    ('NECK', 'CLAVICLE_RIGHT'),
    ('CLAVICLE_RIGHT', 'SHOULDER_RIGHT'),
    ('SHOULDER_RIGHT', 'ELBOW_RIGHT'),
    ('ELBOW_RIGHT', 'WRIST_RIGHT'),
    ('WRIST_RIGHT', 'HAND_RIGHT'),
    ('HAND_RIGHT', 'HANDTIP_RIGHT'),
    ('HAND_RIGHT', 'THUMB_RIGHT'),
    ('PELVIS', 'HIP_LEFT'),
    ('HIP_LEFT', 'KNEE_LEFT'),
    ('KNEE_LEFT', 'ANKLE_LEFT'),
    ('ANKLE_LEFT', 'FOOT_LEFT'),
    ('PELVIS', 'HIP_RIGHT'),
    ('HIP_RIGHT', 'KNEE_RIGHT'),
    ('KNEE_RIGHT', 'ANKLE_RIGHT'),
    ('ANKLE_RIGHT', 'FOOT_RIGHT'),
    ('HEAD', 'NOSE'),
    ('NOSE', 'EYE_LEFT'),
    ('EYE_LEFT', 'EAR_LEFT'),
    ('NOSE', 'EYE_RIGHT'),
    ('EYE_RIGHT', 'EAR_RIGHT'),
]


# ─── Helper: load one sample ─────────────────────────────────────────
def find_first_session(root):
    """Return the first valid session path under *root*."""
    for batch in sorted(os.listdir(root)):
        batch_dir = os.path.join(root, batch)
        if not os.path.isdir(batch_dir):
            continue
        for session in sorted(os.listdir(batch_dir)):
            session_dir = os.path.join(batch_dir, session)
            csv_path = os.path.join(session_dir, 'synced_data.csv')
            ann_dir = os.path.join(session_dir, 'ego_dataset', 'annotations')
            if os.path.isfile(csv_path) and os.path.isdir(ann_dir):
                return session_dir
    return None


def load_sample(session_dir, frame_idx=0):
    """Load annotation JSON + CSV row for *frame_idx*."""
    ann_path = os.path.join(
        session_dir, 'ego_dataset', 'annotations',
        f'frame_{frame_idx:06d}.json')
    with open(ann_path) as f:
        ann = json.load(f)

    csv_path = os.path.join(session_dir, 'synced_data.csv')
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['frame']) == frame_idx:
                return ann, row
    return ann, None


def preprocess_hmd_data(head, right_hand, left_hand):
    """Replicate _preprocess_hmd_data from the dataset class."""
    midpoint = (right_hand + left_hand) / 2.0
    z_axis = midpoint - head
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm if z_norm > 1e-6 else np.array([0., 0., 1.])

    hand_vector = right_hand - left_hand
    x_axis = np.cross(z_axis, hand_vector)
    x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm if x_norm > 1e-6 else np.array([1., 0., 0.])

    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))

    right_local = R.T @ (right_hand - head)
    left_local = R.T @ (left_hand - head)

    hand_dist = np.linalg.norm(right_local - left_local)
    right_dist = np.linalg.norm(right_local)
    left_dist = np.linalg.norm(left_local)

    return right_local, left_local, hand_dist, right_dist, left_dist, R


# ─── FIGURE 1: Joint Mapping (Kinect 32 → xRegopose 16) ──────────────
def plot_joint_mapping(ann, session_name, frame_idx):
    """Side-by-side 3D: full Kinect skeleton vs mapped xRegopose skeleton."""
    skel3d = {j['name']: j for j in ann['skeleton_3d']}

    # Build Kinect 32 joint positions (mm → m)
    kinect_names = [j['name'] for j in ann['skeleton_3d']]
    kinect_pos = {}
    kinect_conf = {}
    for j in ann['skeleton_3d']:
        kinect_pos[j['name']] = np.array([j['x'], j['y'], j['z']]) / 1000.0
        kinect_conf[j['name']] = j.get('confidence', 0)

    # Build xRegopose 16 joint positions
    xr_pos = np.zeros((16, 3))
    xr_vis = np.zeros(16)
    for i, kname in enumerate(KINECT_TO_XREGOPOSE):
        if kname in skel3d:
            j = skel3d[kname]
            xr_pos[i] = [j['x'], j['y'], j['z']]
            xr_vis[i] = 1.0 if j.get('confidence', 0) >= 2 else 0.0
    xr_pos /= 1000.0  # mm → m

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f'Joint Mapping: Kinect 32 → xRegopose 16\n'
                 f'Session: {session_name}  |  Frame: {frame_idx}',
                 fontsize=16, fontweight='bold')

    # ── LEFT: Full Kinect 32-joint skeleton ──
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Azure Kinect (32 joints)', fontsize=14)

    # Color: selected=red, unselected=gray
    selected_set = set(KINECT_TO_XREGOPOSE)
    for name, pos in kinect_pos.items():
        conf = kinect_conf[name]
        if conf < 2:
            continue  # skip unreliable joints
        is_selected = name in selected_set
        color = '#E74C3C' if is_selected else '#95A5A6'
        size = 60 if is_selected else 25
        ax1.scatter(*pos, c=color, s=size, zorder=5,
                    edgecolors='black' if is_selected else 'none',
                    linewidths=1.5 if is_selected else 0)
        # Label selected joints
        if is_selected:
            xr_idx = KINECT_TO_XREGOPOSE.index(name)
            label = f'{name}\n→ [{xr_idx}] {XREGOPOSE_NAMES[xr_idx]}'
            ax1.text(pos[0], pos[1], pos[2] + 0.04, label,
                     fontsize=6, ha='center', va='bottom',
                     color='#C0392B', fontweight='bold')

    # Draw Kinect links
    for a, b in KINECT_32_LINKS:
        if a in kinect_pos and b in kinect_pos:
            if kinect_conf.get(a, 0) < 2 or kinect_conf.get(b, 0) < 2:
                continue
            pa, pb = kinect_pos[a], kinect_pos[b]
            ax1.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                     c='#BDC3C7', lw=1.0, alpha=0.6)

    _style_3d_axis(ax1, kinect_pos, valid_only=True, conf=kinect_conf)

    # ── RIGHT: Mapped xRegopose 16-joint skeleton ──
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('xRegopose (16 mapped joints)', fontsize=14)

    # Color by body part
    colors_upper = '#3498DB'  # blue
    colors_lower = '#E67E22'  # orange
    colors_center = '#2ECC71'  # green

    for i in range(16):
        if xr_vis[i] < 1:
            continue
        if i <= 1:
            c = colors_center
        elif i <= 7:
            c = colors_upper
        else:
            c = colors_lower
        ax2.scatter(*xr_pos[i], c=c, s=70, zorder=5,
                    edgecolors='black', linewidths=1)
        ax2.text(xr_pos[i, 0], xr_pos[i, 1], xr_pos[i, 2] + 0.04,
                 f'[{i}] {XREGOPOSE_NAMES[i]}',
                 fontsize=7, ha='center', va='bottom', fontweight='bold')

    # Draw xRegopose skeleton links
    link_colors = {
        'center': '#2ECC71',
        'left_arm': '#3498DB',
        'right_arm': '#E67E22',
        'left_leg': '#3498DB',
        'right_leg': '#E67E22',
    }
    for a, b in SKELETON_LINKS:
        if xr_vis[a] < 1 or xr_vis[b] < 1:
            continue
        # Determine link color
        if a == 0 and b == 1:
            lc = link_colors['center']
        elif b in [2, 3, 4] or a in [2, 3]:
            lc = link_colors['left_arm']
        elif b in [5, 6, 7] or a in [5, 6]:
            lc = link_colors['right_arm']
        elif b in [8, 9, 10, 11] or a in [8, 9, 10]:
            lc = link_colors['left_leg']
        else:
            lc = link_colors['right_leg']
        ax2.plot([xr_pos[a, 0], xr_pos[b, 0]],
                 [xr_pos[a, 1], xr_pos[b, 1]],
                 [xr_pos[a, 2], xr_pos[b, 2]],
                 c=lc, lw=2.5, alpha=0.8)

    _style_3d_axis(ax2, {f'{i}': xr_pos[i] for i in range(16) if xr_vis[i]},
                   valid_only=False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Selected for mapping (16)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95A5A6',
               markersize=8, label='Unused Kinect joints'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
               markersize=10, label='Center (Spine2, Head)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
               markersize=10, label='Left side / Upper'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22',
               markersize=10, label='Right side / Lower'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5,
               fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = os.path.join(OUT_DIR, 'fig1_joint_mapping.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)

    return xr_pos, xr_vis


# ─── FIGURE 2: Joint Mapping Table ────────────────────────────────────
def plot_mapping_table(ann):
    """Create a table image showing the 32→16 mapping clearly."""
    skel3d = {j['name']: j for j in ann['skeleton_3d']}
    all_kinect = [j['name'] for j in ann['skeleton_3d']]

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')
    fig.suptitle('Kinect 32 → xRegopose 16 Joint Mapping Table',
                 fontsize=16, fontweight='bold', y=0.98)

    # Table data
    headers = ['Kinect Joint (32)', 'Conf', '→', 'xRegopose Idx', 'xRegopose Name', 'Body Part']
    rows = []
    selected_set = set(KINECT_TO_XREGOPOSE)

    for kname in all_kinect:
        conf = skel3d[kname].get('confidence', 0)
        if kname in selected_set:
            xr_idx = KINECT_TO_XREGOPOSE.index(kname)
            part = 'Center' if xr_idx <= 1 else ('Upper' if xr_idx <= 7 else 'Lower')
            rows.append([kname, str(conf), '→', str(xr_idx),
                         XREGOPOSE_NAMES[xr_idx], part])
        else:
            rows.append([kname, str(conf), '', '—', '(not used)', ''])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            kname = all_kinect[row - 1]
            if kname in selected_set:
                cell.set_facecolor('#FADBD8')  # light red for selected
            else:
                cell.set_facecolor('#F2F3F4')  # light gray for unused

    out_path = os.path.join(OUT_DIR, 'fig2_mapping_table.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ─── FIGURE 3: HMD Data Visualization ─────────────────────────────────
def plot_hmd_data(ann, csv_row, xr_pos, xr_vis, session_name, frame_idx):
    """Show HMD sensor positions (world) overlaid on skeleton, plus the
    local-coordinate HMD info vector used as model input."""
    # Parse HMD positions (meters) from CSV
    head = np.array([float(csv_row['hmd_pos_x']),
                     float(csv_row['hmd_pos_y']),
                     float(csv_row['hmd_pos_z'])])
    left_hand = np.array([float(csv_row['left_pos_x']),
                          float(csv_row['left_pos_y']),
                          float(csv_row['left_pos_z'])])
    right_hand = np.array([float(csv_row['right_pos_x']),
                           float(csv_row['right_pos_y']),
                           float(csv_row['right_pos_z'])])

    right_local, left_local, hand_dist, right_dist, left_dist, R = \
        preprocess_hmd_data(head, right_hand, left_hand)

    fig = plt.figure(figsize=(26, 11))
    fig.suptitle(f'HMD Data Integration\n'
                 f'Session: {session_name}  |  Frame: {frame_idx}',
                 fontsize=16, fontweight='bold')

    # ── LEFT: World-space skeleton + HMD positions ──
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('World Space: Skeleton + HMD Sensors', fontsize=12)

    # Draw xRegopose skeleton (faded)
    for i in range(16):
        if xr_vis[i] < 1:
            continue
        ax1.scatter(*xr_pos[i], c='#BDC3C7', s=30, alpha=0.5, zorder=3)
    for a, b in SKELETON_LINKS:
        if xr_vis[a] < 1 or xr_vis[b] < 1:
            continue
        ax1.plot([xr_pos[a, 0], xr_pos[b, 0]],
                 [xr_pos[a, 1], xr_pos[b, 1]],
                 [xr_pos[a, 2], xr_pos[b, 2]],
                 c='#BDC3C7', lw=1.5, alpha=0.4)

    # HMD positions (large, prominent)
    ax1.scatter(*head, c='#E74C3C', s=250, marker='D', zorder=10,
                edgecolors='black', linewidths=2, label='HMD (Head)')
    ax1.scatter(*left_hand, c='#3498DB', s=200, marker='^', zorder=10,
                edgecolors='black', linewidths=2, label='Left Controller')
    ax1.scatter(*right_hand, c='#E67E22', s=200, marker='s', zorder=10,
                edgecolors='black', linewidths=2, label='Right Controller')

    # Lines from head to hands
    ax1.plot([head[0], left_hand[0]], [head[1], left_hand[1]],
             [head[2], left_hand[2]], '--', c='#3498DB', lw=2, alpha=0.7)
    ax1.plot([head[0], right_hand[0]], [head[1], right_hand[1]],
             [head[2], right_hand[2]], '--', c='#E67E22', lw=2, alpha=0.7)
    ax1.plot([left_hand[0], right_hand[0]], [left_hand[1], right_hand[1]],
             [left_hand[2], right_hand[2]], ':', c='#9B59B6', lw=2, alpha=0.7)

    # Midpoint
    mid = (right_hand + left_hand) / 2.0
    ax1.scatter(*mid, c='#9B59B6', s=80, marker='x', zorder=8)

    # Annotate positions
    ax1.text(head[0], head[1], head[2]+0.08,
             f'HMD\n({head[0]:.3f}, {head[1]:.3f}, {head[2]:.3f})',
             fontsize=7, ha='center', color='#C0392B', fontweight='bold')
    ax1.text(left_hand[0], left_hand[1], left_hand[2]+0.08,
             f'L-Ctrl\n({left_hand[0]:.3f}, {left_hand[1]:.3f}, {left_hand[2]:.3f})',
             fontsize=7, ha='center', color='#2980B9', fontweight='bold')
    ax1.text(right_hand[0], right_hand[1], right_hand[2]+0.08,
             f'R-Ctrl\n({right_hand[0]:.3f}, {right_hand[1]:.3f}, {right_hand[2]:.3f})',
             fontsize=7, ha='center', color='#D35400', fontweight='bold')

    ax1.legend(fontsize=9, loc='upper left')
    _style_3d_axis(ax1,
                   {'h': head, 'l': left_hand, 'r': right_hand,
                    **{str(i): xr_pos[i] for i in range(16) if xr_vis[i]}},
                   valid_only=False)

    # ── MIDDLE: Local coordinate system visualization ──
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Head-Centric Local Coordinates\n(HMD Preprocessing)', fontsize=12)

    # Origin = head
    origin = np.array([0, 0, 0])
    ax2.scatter(*origin, c='#E74C3C', s=200, marker='D', zorder=10,
                edgecolors='black', linewidths=2, label='Head (Origin)')

    # Local hand positions
    ax2.scatter(*right_local, c='#E67E22', s=150, marker='s', zorder=10,
                edgecolors='black', linewidths=2, label='Right (local)')
    ax2.scatter(*left_local, c='#3498DB', s=150, marker='^', zorder=10,
                edgecolors='black', linewidths=2, label='Left (local)')

    # Lines
    ax2.plot([0, right_local[0]], [0, right_local[1]], [0, right_local[2]],
             '--', c='#E67E22', lw=2, alpha=0.7)
    ax2.plot([0, left_local[0]], [0, left_local[1]], [0, left_local[2]],
             '--', c='#3498DB', lw=2, alpha=0.7)
    ax2.plot([left_local[0], right_local[0]],
             [left_local[1], right_local[1]],
             [left_local[2], right_local[2]],
             ':', c='#9B59B6', lw=2, alpha=0.7)

    # Draw local axes
    axis_len = 0.3
    ax2.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.15, lw=2)
    ax2.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.15, lw=2)
    ax2.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.15, lw=2)
    ax2.text(axis_len+0.02, 0, 0, 'X', fontsize=10, color='red', fontweight='bold')
    ax2.text(0, axis_len+0.02, 0, 'Y', fontsize=10, color='green', fontweight='bold')
    ax2.text(0, 0, axis_len+0.02, 'Z (→midpoint)', fontsize=10, color='blue', fontweight='bold')

    # Annotate
    ax2.text(right_local[0], right_local[1], right_local[2]+0.05,
             f'R: ({right_local[0]:.3f}, {right_local[1]:.3f}, {right_local[2]:.3f})',
             fontsize=8, ha='center', color='#D35400', fontweight='bold')
    ax2.text(left_local[0], left_local[1], left_local[2]+0.05,
             f'L: ({left_local[0]:.3f}, {left_local[1]:.3f}, {left_local[2]:.3f})',
             fontsize=8, ha='center', color='#2980B9', fontweight='bold')

    ax2.legend(fontsize=9, loc='upper left')
    all_pts = {'o': origin, 'r': right_local, 'l': left_local}
    _style_3d_axis(ax2, all_pts, valid_only=False, pad=0.15)

    # ── RIGHT: HMD info 9-dim vector breakdown ──
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.set_title('9-dim HMD Info Vector\n(Model Input)', fontsize=12)

    hmd_info = np.concatenate([
        right_local, left_local,
        [hand_dist, right_dist, left_dist]
    ])

    # Build table
    dim_labels = [
        '[0] Right local X', '[1] Right local Y', '[2] Right local Z',
        '[3] Left local X', '[4] Left local Y', '[5] Left local Z',
        '[6] Hand distance', '[7] Right distance', '[8] Left distance',
    ]
    dim_descriptions = [
        'Right hand X in head frame',
        'Right hand Y in head frame',
        'Right hand Z in head frame',
        'Left hand X in head frame',
        'Left hand Y in head frame',
        'Left hand Z in head frame',
        '‖right_local − left_local‖',
        '‖right_local‖ (head→R)',
        '‖left_local‖ (head→L)',
    ]
    cell_data = [[dl, f'{v:.4f}', desc]
                 for dl, v, desc in zip(dim_labels, hmd_info, dim_descriptions)]

    tbl = ax3.table(
        cellText=cell_data,
        colLabels=['Dimension', 'Value', 'Description'],
        cellLoc='center', loc='center',
        colWidths=[0.28, 0.18, 0.54])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    # Style
    colors_row = ['#FADBD8'] * 3 + ['#D6EAF8'] * 3 + ['#D5F5E3'] * 3
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(colors_row[row - 1])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(OUT_DIR, 'fig3_hmd_data.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ─── FIGURE 4: HMD preprocessing pipeline diagram ─────────────────────
def plot_hmd_pipeline(csv_row):
    """Diagram showing the HMD data flow from CSV → 9-dim vector."""
    head = np.array([float(csv_row['hmd_pos_x']),
                     float(csv_row['hmd_pos_y']),
                     float(csv_row['hmd_pos_z'])])
    left_hand = np.array([float(csv_row['left_pos_x']),
                          float(csv_row['left_pos_y']),
                          float(csv_row['left_pos_z'])])
    right_hand = np.array([float(csv_row['right_pos_x']),
                           float(csv_row['right_pos_y']),
                           float(csv_row['right_pos_z'])])

    right_local, left_local, hand_dist, right_dist, left_dist, R = \
        preprocess_hmd_data(head, right_hand, left_hand)

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.suptitle('HMD Data Preprocessing Pipeline\n'
                 '(synced_data.csv → 9-dim HMD Info)',
                 fontsize=18, fontweight='bold', y=0.97)

    # ── Stage 1: Raw CSV input ──
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                     edgecolor='#2980B9', linewidth=2)
    ax.text(1.5, 9.0, 'Stage 1: Raw CSV Input (synced_data.csv)',
            fontsize=14, fontweight='bold', color='#2980B9')

    csv_text = (
        f'hmd_pos:   ({head[0]:.4f}, {head[1]:.4f}, {head[2]:.4f}) m\n'
        f'left_pos:  ({left_hand[0]:.4f}, {left_hand[1]:.4f}, {left_hand[2]:.4f}) m\n'
        f'right_pos: ({right_hand[0]:.4f}, {right_hand[1]:.4f}, {right_hand[2]:.4f}) m'
    )
    ax.text(2.0, 8.2, csv_text, fontsize=11, fontfamily='monospace',
            bbox=box_props, verticalalignment='top')

    # Arrow
    ax.annotate('', xy=(5, 7.0), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='#7F8C8D'))

    # ── Stage 2: Local coordinate system ──
    box_props2 = dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7',
                      edgecolor='#F39C12', linewidth=2)
    ax.text(1.5, 6.8, 'Stage 2: Build Local Coordinate System',
            fontsize=14, fontweight='bold', color='#F39C12')

    mid = (right_hand + left_hand) / 2.0
    coord_text = (
        f'midpoint = (left + right) / 2 = ({mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f})\n'
        f'Z-axis  = normalize(midpoint − head)     [head → hand midpoint]\n'
        f'X-axis  = normalize(Z × (right − left))  [perpendicular]\n'
        f'Y-axis  = Z × X                          [complete frame]\n'
        f'R = [X | Y | Z]   (3×3 rotation matrix)'
    )
    ax.text(2.0, 6.0, coord_text, fontsize=10, fontfamily='monospace',
            bbox=box_props2, verticalalignment='top')

    # Arrow
    ax.annotate('', xy=(5, 4.8), xytext=(5, 5.3),
                arrowprops=dict(arrowstyle='->', lw=3, color='#7F8C8D'))

    # ── Stage 3: Transform to local ──
    box_props3 = dict(boxstyle='round,pad=0.5', facecolor='#FDEDEC',
                      edgecolor='#E74C3C', linewidth=2)
    ax.text(1.5, 4.6, 'Stage 3: Transform to Head-Local Coordinates',
            fontsize=14, fontweight='bold', color='#E74C3C')

    xform_text = (
        f'right_local = R^T · (right − head) = ({right_local[0]:.4f}, {right_local[1]:.4f}, {right_local[2]:.4f})\n'
        f'left_local  = R^T · (left  − head) = ({left_local[0]:.4f}, {left_local[1]:.4f}, {left_local[2]:.4f})'
    )
    ax.text(2.0, 3.9, xform_text, fontsize=10, fontfamily='monospace',
            bbox=box_props3, verticalalignment='top')

    # Arrow
    ax.annotate('', xy=(5, 2.9), xytext=(5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='#7F8C8D'))

    # ── Stage 4: Compute distances ──
    box_props4 = dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5',
                      edgecolor='#1ABC9C', linewidth=2)
    ax.text(1.5, 2.7, 'Stage 4: Compute Distances & Assemble 9-dim Vector',
            fontsize=14, fontweight='bold', color='#1ABC9C')

    dist_text = (
        f'hand_distance  = ‖right_local − left_local‖ = {hand_dist:.4f} m\n'
        f'right_distance = ‖right_local‖               = {right_dist:.4f} m\n'
        f'left_distance  = ‖left_local‖                = {left_dist:.4f} m\n'
        f'\n'
        f'hmd_info[9] = [right_local(3), left_local(3), hand_dist, right_dist, left_dist]'
    )
    ax.text(2.0, 2.0, dist_text, fontsize=10, fontfamily='monospace',
            bbox=box_props4, verticalalignment='top')

    hmd_info = np.concatenate([right_local, left_local,
                               [hand_dist, right_dist, left_dist]])
    ax.text(2.0, 0.5,
            'Final: [' + ', '.join(f'{v:.4f}' for v in hmd_info) + ']',
            fontsize=12, fontfamily='monospace', fontweight='bold',
            color='#1ABC9C',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#A3E4D7',
                      edgecolor='#1ABC9C', linewidth=2))

    out_path = os.path.join(OUT_DIR, 'fig4_hmd_pipeline.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ─── FIGURE 5: Multiple frames comparison ─────────────────────────────
def plot_multi_frame(session_dir, session_name, frame_indices):
    """Show 3D skeleton + HMD for several frames side by side."""
    n = len(frame_indices)
    fig = plt.figure(figsize=(8 * n, 9))
    fig.suptitle(f'Multiple Frames: Skeleton + HMD\nSession: {session_name}',
                 fontsize=16, fontweight='bold')

    for idx, fi in enumerate(frame_indices):
        try:
            ann, csv_row = load_sample(session_dir, fi)
        except Exception:
            continue
        if csv_row is None or ann.get('num_bodies', 0) == 0:
            continue

        skel3d = {j['name']: j for j in ann['skeleton_3d']}
        xr_pos = np.zeros((16, 3))
        xr_vis = np.zeros(16)
        for i, kname in enumerate(KINECT_TO_XREGOPOSE):
            if kname in skel3d:
                j = skel3d[kname]
                xr_pos[i] = [j['x'] / 1000, j['y'] / 1000, j['z'] / 1000]
                xr_vis[i] = 1.0 if j.get('confidence', 0) >= 2 else 0.0

        head = np.array([float(csv_row['hmd_pos_x']),
                         float(csv_row['hmd_pos_y']),
                         float(csv_row['hmd_pos_z'])])
        left_hand = np.array([float(csv_row['left_pos_x']),
                              float(csv_row['left_pos_y']),
                              float(csv_row['left_pos_z'])])
        right_hand = np.array([float(csv_row['right_pos_x']),
                               float(csv_row['right_pos_y']),
                               float(csv_row['right_pos_z'])])

        ax = fig.add_subplot(1, n, idx + 1, projection='3d')
        ax.set_title(f'Frame {fi}', fontsize=12)

        # Skeleton
        for i in range(16):
            if xr_vis[i] < 1:
                continue
            c = '#2ECC71' if i <= 1 else ('#3498DB' if i <= 7 else '#E67E22')
            ax.scatter(*xr_pos[i], c=c, s=40, zorder=5)
        for a, b in SKELETON_LINKS:
            if xr_vis[a] < 1 or xr_vis[b] < 1:
                continue
            ax.plot([xr_pos[a, 0], xr_pos[b, 0]],
                    [xr_pos[a, 1], xr_pos[b, 1]],
                    [xr_pos[a, 2], xr_pos[b, 2]],
                    c='#7F8C8D', lw=1.5, alpha=0.6)

        # HMD
        ax.scatter(*head, c='#E74C3C', s=180, marker='D', zorder=10,
                   edgecolors='black', linewidths=1.5)
        ax.scatter(*left_hand, c='#3498DB', s=120, marker='^', zorder=10,
                   edgecolors='black', linewidths=1.5)
        ax.scatter(*right_hand, c='#E67E22', s=120, marker='s', zorder=10,
                   edgecolors='black', linewidths=1.5)

        ax.plot([head[0], left_hand[0]], [head[1], left_hand[1]],
                [head[2], left_hand[2]], '--', c='#3498DB', lw=1.5, alpha=0.6)
        ax.plot([head[0], right_hand[0]], [head[1], right_hand[1]],
                [head[2], right_hand[2]], '--', c='#E67E22', lw=1.5, alpha=0.6)

        _style_3d_axis(ax,
                       {'h': head, 'l': left_hand, 'r': right_hand,
                        **{str(i): xr_pos[i] for i in range(16) if xr_vis[i]}},
                       valid_only=False)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C',
               markersize=12, label='HMD (Head)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498DB',
               markersize=12, label='Left Controller'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E67E22',
               markersize=12, label='Right Controller'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_path = os.path.join(OUT_DIR, 'fig5_multi_frame.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ─── FIGURE 6: Ego image + overlaid 2D keypoints ──────────────────────
def plot_ego_image_with_keypoints(ann, session_dir, frame_idx, session_name):
    """Show the egocentric image with 2D skeleton overlay."""
    img_path = os.path.join(
        session_dir, 'ego_dataset', 'images',
        f'frame_{frame_idx:06d}.jpg')
    if not os.path.isfile(img_path):
        print(f'Image not found: {img_path}')
        return

    img = plt.imread(img_path)
    skel2d = {j['name']: j for j in ann['skeleton_2d']}
    skel3d = {j['name']: j for j in ann['skeleton_3d']}

    p2d = np.zeros((16, 2))
    vis = np.zeros(16)
    for i, kname in enumerate(KINECT_TO_XREGOPOSE):
        if kname in skel2d and kname in skel3d:
            j2d = skel2d[kname]
            conf = skel3d[kname].get('confidence', 0)
            p2d[i] = [j2d['u'], j2d['v']]
            vis[i] = 1.0 if conf >= 2 else 0.0

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img)
    ax.set_title(f'Egocentric View + 2D Keypoints\n{session_name} / Frame {frame_idx}',
                 fontsize=14, fontweight='bold')

    # Draw links
    for a, b in SKELETON_LINKS:
        if vis[a] < 1 or vis[b] < 1:
            continue
        if b in [2, 3, 4] or a in [2, 3]:
            lc = '#3498DB'
        elif b in [5, 6, 7] or a in [5, 6]:
            lc = '#E67E22'
        elif b in [8, 9, 10, 11] or a in [8, 9, 10]:
            lc = '#3498DB'
        elif b in [12, 13, 14, 15] or a in [12, 13, 14]:
            lc = '#E67E22'
        else:
            lc = '#2ECC71'
        ax.plot([p2d[a, 0], p2d[b, 0]], [p2d[a, 1], p2d[b, 1]],
                c=lc, lw=2.5, alpha=0.8)

    # Draw joints
    for i in range(16):
        if vis[i] < 1:
            continue
        c = '#2ECC71' if i <= 1 else ('#3498DB' if i <= 7 else '#E67E22')
        ax.scatter(p2d[i, 0], p2d[i, 1], c=c, s=60, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.text(p2d[i, 0] + 5, p2d[i, 1] - 5, f'{XREGOPOSE_NAMES[i]}',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=c, alpha=0.7))

    ax.axis('off')
    out_path = os.path.join(OUT_DIR, 'fig6_ego_image.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ─── Axis styling helper ──────────────────────────────────────────────
def _style_3d_axis(ax, positions, valid_only=True, conf=None, pad=0.2):
    """Set equal aspect ratio and labels for 3D axes."""
    pts = []
    for name, pos in positions.items():
        if conf is not None and valid_only:
            c = conf.get(name, 0)
            if c < 2:
                continue
        pts.append(pos)
    if not pts:
        return
    pts = np.array(pts)
    center = pts.mean(axis=0)
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0 + pad

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.tick_params(labelsize=7)


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    session_dir = find_first_session(DATA_ROOT)
    if session_dir is None:
        print('ERROR: No valid session found')
        sys.exit(1)

    session_name = os.path.basename(session_dir)
    print(f'Using session: {session_dir}')

    # Find a frame with bodies
    frame_idx = 0
    ann, csv_row = load_sample(session_dir, frame_idx)
    while ann.get('num_bodies', 0) == 0 or csv_row is None:
        frame_idx += 1
        ann, csv_row = load_sample(session_dir, frame_idx)
        if frame_idx > 100:
            print('ERROR: no usable frame in first 100')
            sys.exit(1)

    print(f'Using frame: {frame_idx}')

    # Fig 1: Joint mapping (Kinect 32 → xRegopose 16)
    print('\n--- Figure 1: Joint Mapping ---')
    xr_pos, xr_vis = plot_joint_mapping(ann, session_name, frame_idx)

    # Fig 2: Mapping table
    print('\n--- Figure 2: Mapping Table ---')
    plot_mapping_table(ann)

    # Fig 3: HMD data
    print('\n--- Figure 3: HMD Data ---')
    plot_hmd_data(ann, csv_row, xr_pos, xr_vis, session_name, frame_idx)

    # Fig 4: HMD pipeline
    print('\n--- Figure 4: HMD Pipeline ---')
    plot_hmd_pipeline(csv_row)

    # Fig 5: Multiple frames
    print('\n--- Figure 5: Multi-Frame ---')
    plot_multi_frame(session_dir, session_name, [0, 10, 20, 30])

    # Fig 6: Ego image with 2D keypoints
    print('\n--- Figure 6: Ego Image ---')
    plot_ego_image_with_keypoints(ann, session_dir, frame_idx, session_name)

    print(f'\nAll figures saved to: {OUT_DIR}')
