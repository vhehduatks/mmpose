"""Visualize the Cascaded Refinement + Both From Ground V2b Architecture.

Generates a publication-quality architecture diagram showing:
- Stage 1: Coarse Pose Estimation (ResNet -> Heatmaps -> Encoder -> Decoders)
- Stage 2: Per-Joint Refinement (Feature Assembly -> RefinementMLP -> Residual)

Usage:
    python my_code/visualization/visualize_cascaded_architecture.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ── Colors ─────────────────────────────────────────────────────────────────
C = {
    'input':    '#E3F2FD', 'backbone': '#BBDEFB', 'deconv':  '#90CAF9',
    'heatmap':  '#64B5F6', 'encoder':  '#FFF9C4', 'hmd':     '#FFE0B2',
    'latent':   '#C8E6C9', 'decoder':  '#A5D6A7', 'stage2':  '#E1BEE7',
    'refine':   '#CE93D8', 'output':   '#F8BBD0', 'loss':    '#FFCDD2',
    's1_bg':    '#FAFAFA', 's2_bg':    '#F5F5F5', 'arrow':   '#424242',
    'text':     '#212121', 'skip':     '#1E88E5', 'loss_a':  '#E53935',
    'dim':      '#9E9E9E',
}


def box(ax, cx, cy, w, h, label, color, bold=False, sub=None,
        fs=9, sfs=7, ec='#9E9E9E', lw=0.8):
    """Draw centered rounded box. cx, cy = center. Returns (cx, top, bot)."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.012',
        facecolor=color, edgecolor=ec, linewidth=lw))
    ty = cy + h * 0.12 if sub else cy
    ax.text(cx, ty, label, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=C['text'])
    if sub:
        ax.text(cx, cy - h * 0.22, sub, ha='center', va='center',
                fontsize=sfs, color='#757575', style='italic')
    return cx, cy + h / 2, cy - h / 2


def arr(ax, x1, y1, x2, y2, color=None, lw=1.2, cs='arc3,rad=0', ms=10):
    """Draw arrow from (x1,y1) to (x2,y2)."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='->', color=color or C['arrow'],
        linewidth=lw, connectionstyle=cs, mutation_scale=ms))


def stage_bg(ax, cx, cy, w, h, label, color):
    """Draw stage background with label at top-left."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.006',
        facecolor=color, edgecolor='#BDBDBD', linewidth=1.2, alpha=0.45))
    ax.text(x + 0.01, y + h - 0.006, label, ha='left', va='top',
            fontsize=10, fontweight='bold', color='#757575')


def dim_label(ax, cx, y, text):
    """Small dim-colored shape label."""
    ax.text(cx, y, text, ha='center', va='center',
            fontsize=6.5, color=C['dim'], style='italic')


def main():
    fig, ax = plt.subplots(figsize=(16, 26))
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layout constants
    BW = 0.20   # standard box width
    BH = 0.018  # standard box height
    GAP = 0.006  # vertical gap between boxes
    LEFT = 0.18  # left column center x
    MID = 0.50   # middle center x
    RIGHT = 0.78  # right column center x

    # ══════════════════════════════════════════════════════════════════
    #  TITLE
    # ══════════════════════════════════════════════════════════════════
    ax.text(0.50, 0.990, 'Cascaded Refinement + Both From Ground V2b',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.50, 0.977,
            'Full Body: 34.24mm   |   Upper: 22.04mm   |   Lower: 46.45mm   |   48.8M params',
            ha='center', va='top', fontsize=9.5, color='#757575')

    # ══════════════════════════════════════════════════════════════════
    #  STAGE BACKGROUNDS
    # ══════════════════════════════════════════════════════════════════
    stage_bg(ax, 0.50, 0.720, 0.94, 0.47,
             'STAGE 1: Coarse Pose Estimation', C['s1_bg'])
    stage_bg(ax, 0.50, 0.195, 0.94, 0.37,
             'STAGE 2: Per-Joint Refinement', C['s2_bg'])

    # ══════════════════════════════════════════════════════════════════
    #  INPUTS
    # ══════════════════════════════════════════════════════════════════
    y = 0.955
    _, img_top, img_bot = box(ax, LEFT, y, BW, BH,
                              'Input Image', C['input'], bold=True,
                              sub='[B, 3, 256, 256]')

    _, hmd_top, hmd_bot = box(ax, RIGHT, y, 0.30, BH,
                              'Enhanced HMD Info (12-dim)', C['hmd'], bold=True,
                              sub='Base HMD(9) + Ground Heights(3)')

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 - Left column: Image processing pipeline
    # ══════════════════════════════════════════════════════════════════

    # ResNet-101
    y = 0.915
    _, res_top, res_bot = box(ax, LEFT, y, BW, BH,
                              'ResNet-101', C['backbone'], bold=True,
                              sub='COCO pretrained  |  42.5M')
    arr(ax, LEFT, img_bot, LEFT, res_top)
    dim_label(ax, LEFT + 0.13, res_bot - GAP * 0.5, '[B, 2048, 8, 8]')

    # Deconv + Upsample
    y = 0.875
    _, dc_top, dc_bot = box(ax, LEFT, y, BW, BH,
                            'Deconv + Upsample', C['deconv'],
                            sub='3x ConvTranspose2d + Pool(47)')
    arr(ax, LEFT, res_bot - GAP, LEFT, dc_top)

    # Heatmaps
    y = 0.840
    _, hm_top, hm_bot = box(ax, LEFT, y, BW, BH,
                            'Heatmaps', C['heatmap'], bold=True,
                            sub='[B, 16, 47, 47]')
    arr(ax, LEFT, dc_bot, LEFT, hm_top)

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 - Enhanced Encoder (center, wide)
    # ══════════════════════════════════════════════════════════════════
    y_enc = 0.790
    enc_w = 0.56
    _, enc_top, enc_bot = box(ax, MID, y_enc, enc_w, 0.025,
                              'Enhanced Encoder (Heatmap + HMD Fusion)',
                              C['encoder'], bold=True,
                              sub='Conv(16->128) -> GAP -> Lin(128,64)  +  HMD Lin(12,64)  ->  Add',
                              sfs=6.5)
    # heatmap -> encoder
    arr(ax, LEFT + BW / 2 - 0.01, hm_bot, MID - enc_w / 4, enc_top,
        cs='arc3,rad=0.05')
    # HMD -> encoder
    arr(ax, RIGHT, hmd_bot, MID + enc_w / 4, enc_top,
        cs='arc3,rad=-0.12')

    # ── Z latent ───────────────────────────────────────────────────────
    y_z = 0.750
    _, z_top, z_bot = box(ax, MID, y_z, 0.14, 0.016,
                          'Z  [B, 64]', C['latent'], bold=True, fs=9)
    arr(ax, MID, enc_bot, MID, z_top)

    # ── Three decoders from Z ──────────────────────────────────────────
    y_dec = 0.710
    dw = 0.22
    dh = 0.024

    cx_pd, pd_top, pd_bot = box(ax, 0.18, y_dec, dw, dh,
                                'PoseDecoder', C['decoder'], bold=True,
                                sub='Lin(64->512->512->48)  0.5M')
    arr(ax, MID - 0.06, z_bot, 0.18 + 0.03, pd_top, cs='arc3,rad=0.12')

    cx_hd, hd_top, hd_bot = box(ax, MID, y_dec, 0.24, dh,
                                'EfficientHM Decoder', C['decoder'],
                                sub='ConvTranspose upsample  1.35M')
    arr(ax, MID, z_bot, MID, hd_top)

    cx_hr, hr_top, hr_bot = box(ax, 0.80, y_dec, dw, dh,
                                'HMD Reconstructor', C['decoder'],
                                sub='Lin(64->9)  base HMD only')
    arr(ax, MID + 0.06, z_bot, 0.80 - 0.03, hr_top, cs='arc3,rad=-0.12')

    # ── Coarse Pose ────────────────────────────────────────────────────
    y_cp = 0.665
    _, cp_top, cp_bot = box(ax, 0.18, y_cp, dw, 0.02,
                            'Coarse 3D Pose  [B, 16, 3]', C['output'],
                            bold=True, fs=8)
    arr(ax, cx_pd, pd_bot, 0.18, cp_top)

    # ── Stage 1 Losses ─────────────────────────────────────────────────
    y_l1 = 0.660
    l1_w = 0.46
    box(ax, 0.67, y_l1, l1_w, 0.032,
        'Stage 1 Losses', C['loss'], bold=True,
        sub='kpt(1000) + pose(1.0) + cosine(0.1) + limb(0.25) + hm_recon(500) + hmd(1.0)',
        sfs=5.8)
    # loss arrows
    arr(ax, 0.18 + dw / 2, cp_top + 0.003, 0.67 - l1_w / 2, y_l1,
        color=C['loss_a'], lw=0.7)
    arr(ax, cx_hd, hd_bot, 0.60, y_l1 + 0.016,
        color=C['loss_a'], lw=0.7, cs='arc3,rad=-0.05')
    arr(ax, cx_hr, hr_bot, 0.82, y_l1 + 0.016,
        color=C['loss_a'], lw=0.7, cs='arc3,rad=-0.05')

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 -> STAGE 2 Transfer
    # ══════════════════════════════════════════════════════════════════
    y_xfer = 0.505
    xfer_y_label = y_xfer + 0.010
    xfer_y_shape = y_xfer - 0.003
    xfer_y_arrow_bot = y_xfer - 0.018

    transfers = [
        ('Coarse Pose', '[B,16,3]', 0.10),
        ('Heatmaps', '[B,16,47,47]', 0.28),
        ('Backbone', '[B,2048,8,8]', 0.46),
        ('Z Latent', '[B,64]', 0.64),
        ('HMD Info', '[B,12]', 0.82),
    ]
    for label, shape, x in transfers:
        ax.text(x, xfer_y_label, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#757575')
        ax.text(x, xfer_y_shape, shape, ha='center', va='center',
                fontsize=6, color=C['dim'], style='italic')
        arr(ax, x, xfer_y_arrow_bot, x, xfer_y_arrow_bot - 0.012,
            color=C['dim'], lw=1.2, ms=8)

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ══════════════════════════════════════════════════════════════════

    # Row positions for Stage 2 feature extractors
    y_s2_r1 = 0.390  # top row
    y_s2_r2 = 0.355  # second row
    y_s2_r3 = 0.320  # third row
    s2w = 0.17
    s2h = 0.020

    # ── Column 1: Spatial pipeline (Soft-Argmax -> Grid Sampling -> Proj)
    col1 = 0.13
    box(ax, col1, y_s2_r1, s2w, s2h,
        'Soft-Argmax 2D', C['stage2'],
        sub='heatmap -> coords [B,16,2]', fs=7.5, sfs=5.5)
    _, gs_top, gs_bot = box(ax, col1, y_s2_r2, s2w, s2h,
                            'Grid Sampling', C['stage2'], bold=True,
                            sub='F.grid_sample(backbone, coords)', fs=7.5, sfs=5.5)
    arr(ax, col1, y_s2_r1 - s2h / 2, col1, gs_top, lw=0.8)
    _, sp_top, sp_bot = box(ax, col1, y_s2_r3, s2w, s2h,
                            'Spatial Proj', C['stage2'],
                            sub='Lin(2048->64) [B,16,64]', fs=7.5, sfs=5.5)
    arr(ax, col1, gs_bot, col1, sp_top, lw=0.8)

    # ── Column 2: Pose + Kinematic
    col2 = 0.35
    box(ax, col2, y_s2_r1, s2w, s2h,
        'Pose Encoder', C['stage2'],
        sub='flatten -> Lin(48->128)', fs=7.5, sfs=5.5)
    _, kc_top, kc_bot = box(ax, col2, y_s2_r2, s2w, s2h,
                            'Kinematic Chain', C['stage2'],
                            sub='15 bones -> Lin(60->64)', fs=7.5, sfs=5.5)
    arr(ax, col2, y_s2_r1 - s2h / 2, col2, kc_top, lw=0.8)

    # ── Column 3: HMD Encoder S2
    col3 = 0.57
    _, h2_top, h2_bot = box(ax, col3, y_s2_r1, s2w, s2h,
                            'HMD Encoder S2', C['hmd'],
                            sub='Lin(12->32) [B,16,32]', fs=7.5, sfs=5.5)

    # ── Column 4: Z expand
    col4 = 0.78
    _, ze_top, ze_bot = box(ax, col4, y_s2_r1, s2w, s2h,
                            'Z expand', C['latent'],
                            sub='[B,64] -> [B,16,64]', fs=7.5, sfs=5.5)

    # ── Per-Joint Feature Assembly ─────────────────────────────────────
    y_fa = 0.270
    fa_w = 0.80
    _, fa_top, fa_bot = box(
        ax, MID, y_fa, fa_w, 0.025,
        'Per-Joint Feature Assembly  (355-dim per joint)', C['stage2'],
        bold=True,
        sub='cat[ XYZ(3) | Spatial(64) | Z(64) | Pose(128) | Kinematic(64) | HMD(32) ]  ->  [B, 16, 355]',
        sfs=6.5)

    # Arrows from feature extractors into assembly
    for cx_src, y_src_bot in [
        (col1, sp_bot), (col2, kc_bot),
        (col3, h2_bot - s2h / 2), (col4, ze_bot - s2h / 2),
    ]:
        arr(ax, cx_src, y_src_bot, cx_src, fa_top, lw=0.8)

    # ── Feature vector bar ─────────────────────────────────────────────
    y_fv = 0.235
    fv_x_start = 0.14
    fv_w_total = 0.68
    fv_h = 0.013
    parts = [
        ('XYZ', 3, C['output']),
        ('Spatial', 64, '#D1C4E9'),
        ('Z', 64, C['latent']),
        ('Pose', 128, '#E1BEE7'),
        ('Kin', 64, '#D1C4E9'),
        ('HMD', 32, C['hmd']),
    ]
    total_dim = sum(d for _, d, _ in parts)
    cur_x = fv_x_start
    for name, dim, color in parts:
        w = fv_w_total * dim / total_dim
        ax.add_patch(FancyBboxPatch(
            (cur_x, y_fv), w, fv_h, boxstyle='round,pad=0.001',
            facecolor=color, edgecolor='#BDBDBD', linewidth=0.5))
        ax.text(cur_x + w / 2, y_fv + fv_h / 2,
                f'{name}({dim})', ha='center', va='center',
                fontsize=5.5, color=C['text'])
        cur_x += w

    # local vs global bracket
    bnd = fv_x_start + fv_w_total * (3 + 64) / total_dim
    for x0, x1, lbl in [
        (fv_x_start, bnd - 0.002, 'per-joint (local)'),
        (bnd + 0.002, fv_x_start + fv_w_total, 'shared (global)'),
    ]:
        ax.annotate('', xy=(x0, y_fv - 0.004), xytext=(x1, y_fv - 0.004),
                    arrowprops=dict(arrowstyle='<->', color=C['dim'], lw=0.5))
        ax.text((x0 + x1) / 2, y_fv - 0.010, lbl,
                ha='center', fontsize=5.5, color=C['dim'])

    # ── RefinementMLP ──────────────────────────────────────────────────
    y_mlp = 0.185
    _, mlp_top, mlp_bot = box(
        ax, MID, y_mlp, 0.62, 0.025,
        'RefinementMLP  (shared across 16 joints)', C['refine'], bold=True,
        sub='Lin(355->256) -> BN -> ReLU -> [ResBlock(256)]x1 -> Lin(256->3)  ->  delta [B,16,3]',
        sfs=6)
    arr(ax, MID, y_fv, MID, mlp_top)

    # ── Residual Addition ──────────────────────────────────────────────
    y_add = 0.140
    _, add_top, add_bot = box(
        ax, MID, y_add, 0.50, 0.020,
        'Refined Pose  =  Coarse Pose  +  delta', C['output'], bold=True,
        sub='[B, 16, 3]', fs=8.5)
    arr(ax, MID, mlp_bot, MID, add_top)

    # Skip connection (coarse pose to residual add)
    skip_x = 0.05
    # vertical line down
    ax.plot([skip_x, skip_x], [y_xfer - 0.03, y_add],
            color=C['skip'], lw=1.8, ls='--', zorder=3)
    # horizontal line from coarse pose transfer to skip
    arr(ax, 0.10, xfer_y_arrow_bot - 0.012, skip_x, y_xfer - 0.03,
        color=C['skip'], lw=1.8, cs='arc3,rad=0.08')
    # horizontal line from skip to add box
    arr(ax, skip_x, y_add, MID - 0.25, y_add,
        color=C['skip'], lw=1.8)
    ax.text(skip_x - 0.008, 0.32, 'skip\nconnection',
            ha='center', va='center', fontsize=6.5, color=C['skip'],
            fontweight='bold', rotation=90)

    # ── Stage 2 Losses ─────────────────────────────────────────────────
    y_l2 = 0.115
    box(ax, 0.82, y_l2, 0.22, 0.022,
        'Stage 2 Losses', C['loss'], bold=True,
        sub='pose(1.0) + bone(0.5) + sym(0.1)', sfs=5.5)
    arr(ax, MID + 0.25, add_bot, 0.82, y_l2 + 0.011,
        color=C['loss_a'], lw=0.8, cs='arc3,rad=-0.08')

    # ── Final Output ───────────────────────────────────────────────────
    y_out = 0.075
    box(ax, MID, y_out, 0.58, 0.025,
        'OUTPUT:  Refined 3D Pose  [B, 16, 3]', '#E8F5E9', bold=True,
        sub='34.24mm MPJPE   (-17.2% vs baseline 41.37mm)',
        ec='#4CAF50', lw=2)
    arr(ax, MID, add_bot, MID, y_out + 0.025 / 2, lw=1.5)

    # ══════════════════════════════════════════════════════════════════
    #  LEGEND
    # ══════════════════════════════════════════════════════════════════
    legend_items = [
        ('Backbone / Conv', C['backbone']),
        ('Encoder / Fusion', C['encoder']),
        ('HMD Sensor Info', C['hmd']),
        ('Latent Z', C['latent']),
        ('Decoder', C['decoder']),
        ('Stage 2 Module', C['stage2']),
        ('Loss', C['loss']),
        ('Output / Pose', C['output']),
    ]
    patches = [mpatches.Patch(fc=c, ec='#9E9E9E', label=l, lw=0.5)
               for l, c in legend_items]
    ax.legend(handles=patches, loc='upper right', fontsize=7,
              framealpha=0.92, ncol=2, bbox_to_anchor=(0.97, 0.965),
              columnspacing=1.0, handlelength=1.2)

    # ══════════════════════════════════════════════════════════════════
    #  DESIGN ANNOTATIONS (right margin)
    # ══════════════════════════════════════════════════════════════════
    ann_kw = dict(ha='left', va='center', fontsize=6, color='#78909C',
                  bbox=dict(boxstyle='round,pad=0.25', fc='white',
                            ec='#CFD8DC', lw=0.5, alpha=0.9))
    for y_a, txt in [
        (0.790, 'Element-wise Add\n(not concat)\nHMD as residual bias'),
        (0.710, 'EfficientDecoder\n1.35M vs 40M (-96.6%)\nConvTranspose upsample'),
        (0.355, 'Grid Sampling\nsamples AT each joint\npreserves spatial detail'),
        (0.185, 'Residual refinement\nlearn delta correction\nnot full pose'),
    ]:
        ax.text(0.96, y_a, txt, **ann_kw)

    # ── Save ───────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.2)
    out_dir = 'output_architecture_vis'
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'cascaded_v2b_architecture.{ext}')
        plt.savefig(path, dpi=200 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'Saved: {path}')
    plt.close()


if __name__ == '__main__':
    main()
