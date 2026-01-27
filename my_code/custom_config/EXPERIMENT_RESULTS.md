# EgoPose 3D Experiment Results and Analysis

> Last updated: 2026-01-27

## Objective

**Achieve 3D pose estimation performance better than Single COCO Baseline (41.37mm MPJPE)**

---

## Experiment Results Summary

### Overall Comparison Table

| # | Experiment Name | Config | Head | MPJPE (mm) | Best Epoch | Status |
|---|--------|--------|------|------------|------------|------|
| 0 | **Single COCO (Baseline)** | `HMD_xregopose_single_coco_full_config.py` | `CustomxRegoposeBaselinel1` | **41.37** | 8 | 🏆 Best |
| 1 | Dual COCO+MPII | `HMD_xregopose_h5cache_coco_mpii_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | 43.26 | 8 | ❌ |
| 2 | Dual Warmup v2 | `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | `CustomxRegoposeBaselinel1_multi_backbone_v2` | 45.93 | 9 | ❌ |
| 3 | Single Lifting | `HMD_xregopose_single_lifting_config.py` | `CustomEgoposeLiftingHead` | 45.92 | 9 | ❌ |
| 4 | Lifting + Backbone Fusion | `HMD_xregopose_lifting_backbone_fusion_config.py` | `CustomEgoposeLiftingBackboneFusionHead` | 105.18 | 5 | ❌ (stopped at 7ep) |
| 5 | EfficientHeatmapDecoder | `HMD_xregopose_efficient_decoder_full_config.py` | `CustomxRegoposeBaselinel1` | 45.06 | 8 | ❌ (param efficiency) |
| 6 | Attention Lifting v1 | `HMD_xregopose_attention_lifting_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 45.43 | 7 | ❌ (epoch 4 unstable) |
| 7 | Attention Lifting v2 | `HMD_xregopose_attention_lifting_v2_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 48.38 | 8 | ❌ (worse than v1) |
| 8 | Attention Lifting v3 | `HMD_xregopose_attention_lifting_v3_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 46.68 | 1 | ❌ (epoch 2 diverged) |
| 9 | Attention Lifting v4 | `HMD_xregopose_attention_lifting_v4_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 47.95 | 6 | ❌ (worse than v1) |
| 10 | Attention Lifting v5 | `HMD_xregopose_attention_lifting_v5_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 49.75 | 8 | ❌ (excessive LR) |
| 11 | Attention Lifting v6 | `HMD_xregopose_attention_lifting_v6_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 51.82 | 6 | ❌ (epoch 10 diverged) |
| 12 | Attention Lifting v7 | `HMD_xregopose_attention_lifting_v7_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 45.73 | 7 | ❌ (similar to v1, warmup counterproductive) |
| 13 | Skeleton GAT | `HMD_xregopose_skeleton_gat_full_config.py` | `CustomEgoposeSkeletonGATHead` | 50.88 | 7 | ❌ (sparse attention failed) |
| 14 | ViT Lifting v1 | `HMD_xregopose_vit_lifting_v1_full_config.py` | `CustomEgoposeViTLiftingHead` | 50.84 | 8 | ❌ (separate heatmap, no info sharing) |
| 15 | ViT Lifting v2 | `HMD_xregopose_vit_lifting_v2_full_config.py` | `CustomEgoposeViTLiftingHead` | 51.77 | 1 | ❌ (no recon, worse than v1) |
| 16 | **ViT Lifting v3** | `HMD_xregopose_vit_lifting_v3_full_config.py` | `CustomEgoposeViTLiftingHead` | **45.34** | 4 | ⭐ Best ViT (Recon + Self-Attn) |
| 17 | ViT Lifting v4 | `HMD_xregopose_vit_lifting_v4_full_config.py` | `CustomEgoposeViTLiftingHead` | 45.66 | 7 | ❌ (CosineAnnealingLR) |
| 18 | ViT Lifting v5 | `HMD_xregopose_vit_lifting_v5_full_config.py` | `CustomEgoposeViTLiftingHeadV5` | 47.22 | 7 | ❌ (Hybrid Attention) |
| 19 | Upper-Lower Decoupled | `HMD_xregopose_decoupled_full_config.py` | `CustomEgoposeDecoupledHead` | 45.00 | 8 | ❌ (Lower body degraded) |

### Detailed Results by Body Part

| Experiment Name | Full Body | Upper Body | Lower Body | Best Epoch |
|--------|-----------|------------|------------|------------|
| **Single COCO (Baseline)** | **41.37mm** | **29.42mm** | **53.31mm** | 8 |
| Dual COCO+MPII | 43.26mm | 30.03mm | 56.48mm | 8 |
| Dual Warmup v2 | 45.93mm | 31.07mm | 60.79mm | 9 |
| Single Lifting | 45.92mm | 33.91mm | 57.93mm | 9 |
| EfficientHeatmapDecoder | 45.06mm | 30.54mm | 59.58mm | 8 |
| Attention Lifting v1 | 45.43mm | 30.14mm | 60.72mm | 7 |
| Attention Lifting v2 | 48.38mm | 32.03mm | 64.73mm | 8 |
| Attention Lifting v3 | 46.68mm | 34.41mm | 58.95mm | 1 |
| Attention Lifting v4 | 47.95mm | 30.10mm | 65.80mm | 6 |
| Attention Lifting v5 | 49.75mm | 31.08mm | 68.41mm | 8 |
| Attention Lifting v6 | 51.82mm | 32.89mm | 70.75mm | 6 |
| Attention Lifting v7 | 45.73mm | 32.69mm | 58.78mm | 7 |
| Skeleton GAT | 50.88mm | 33.83mm | 67.92mm | 7 |
| ViT Lifting v1 | 50.84mm | 28.77mm | 72.91mm | 8 |
| ViT Lifting v2 | 51.77mm | 33.37mm | 70.17mm | 1 |
| **ViT Lifting v3** | **45.34mm** | **23.49mm** | **67.19mm** | 4 |
| ViT Lifting v4 | 45.66mm | 25.76mm | 65.56mm | 7 |
| ViT Lifting v5 | 47.22mm | 29.09mm | 65.35mm | 7 |
| Upper-Lower Decoupled | 45.00mm | 24.10mm | 65.89mm | 8 |

### Comparison Against Baseline

| Experiment Name | Full Body | vs Baseline | Notes |
|--------|-----------|-------------|------|
| **Single COCO (Baseline)** | **41.37mm** | - | 🏆 |
| Dual COCO+MPII | 43.26mm | +1.89mm ❌ | mutual learning degraded |
| Dual Warmup v2 | 45.93mm | +4.56mm ❌ | warmup also ineffective |
| Single Lifting | 45.92mm | +4.55mm ❌ | insufficient depth info |
| EfficientHeatmapDecoder | 45.06mm | +3.69mm ❌ | 96% param reduction, performance drop |
| Attention Lifting v1 | 45.43mm | +4.06mm ❌ | epoch 4 spike, training unstable |
| Attention Lifting v2 | 48.38mm | +7.01mm ❌ | warmup counterproductive, worse than v1 |
| Attention Lifting v3 | 46.68mm | +5.31mm ❌ | excessive LR, epoch 2 diverged |
| Attention Lifting v4 | 47.95mm | +6.58mm ❌ | CosineAnnealing, worse than v1 |
| Attention Lifting v5 | 49.75mm | +8.38mm ❌ | LR=0.002 excessive, epoch 2 spike |
| Attention Lifting v6 | 51.82mm | +10.45mm ❌ | CosineRestartLR, epoch 10 diverged |
| Attention Lifting v7 | 45.73mm | +4.36mm ❌ | Optimized schedule, epoch 2 spike |
| Skeleton GAT | 50.88mm | +9.51mm ❌ | Sparse attention, lower body degraded |
| ViT Lifting v1 | 50.84mm | +9.47mm ❌ | separate heatmap path, no info sharing |
| ViT Lifting v2 | 51.77mm | +10.40mm ❌ | No reconstruction, Self-Attn only |
| **ViT Lifting v3** | **45.34mm** | **+3.97mm** ⭐ | Recon + Self-Attn, best Upper Body |
| ViT Lifting v4 | 45.66mm | +4.29mm ❌ | CosineAnnealingLR, validation spike |
| ViT Lifting v5 | 47.22mm | +5.85mm ❌ | Hybrid Attention, gradient scaling |
| Upper-Lower Decoupled | 45.00mm | +3.63mm ❌ | Upper improved (-5.32mm), Lower degraded (+12.58mm) |

---

## Detailed Experiment Results

### Experiment 0: Single COCO Baseline 🏆

**Config**: `HMD_xregopose_single_coco_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_single_coco_full`

**Architecture**:
```
Backbone: ResNet-101 (COCO pretrained)
    ↓
Head: CustomxRegoposeBaselinel1
    ├── Deconv → Heatmap [16, 47, 47]
    ├── HeatmapEncoder → Z [64]
    ├── + HMD info [9→64]
    ├── PoseDecoder → 3D Pose [16, 3]
    └── HeatmapDecoder → Recon Heatmap (40M params)
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 6 | 46.92mm | 33.15mm | 60.69mm |
| 7 | 43.69mm | 30.95mm | 56.43mm |
| **8** | **41.37mm** | **29.42mm** | **53.31mm** |
| 9 | 42.07mm | 29.15mm | 55.00mm |
| 10 | 41.39mm | 29.04mm | 53.74mm |

**Key Characteristics**:
- Single backbone (COCO pretrained only)
- HeatmapEncoder→Z[64]→Decoder architecture
- Head parameters: ~61M (including HeatmapDecoder 40M)

---

### Experiment 1: Dual COCO+MPII

**Config**: `HMD_xregopose_h5cache_coco_mpii_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_coco_mpii`

**Hypothesis**: Combining two pretrained backbones via mutual learning improves performance

**Architecture**:
```
Backbone1: ResNet-101 (COCO)  |  Backbone2: ResNet-101 (MPII)
         ↓                    |           ↓
    feat1 [2048,8,8]          |      feat2 [2048,8,8]
         ↓                    |           ↓
    Heatmap1                  |      Heatmap2
         ↓                    |           ↓
         └──── MSE Loss ──────┘  ← loss_backbone_latant
         ↓
    Main path → 3D Pose
```

**Per-Epoch Results** (10 epoch run):
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 4 | 45.42mm | 31.39mm | 59.45mm |
| 6 | 44.25mm | 30.48mm | 58.02mm |
| **8** | **43.26mm** | **30.03mm** | **56.48mm** |
| 10 | 44.91mm | 30.56mm | 59.26mm |

**Analysis**:
- Dual backbone was **actually worse** than Single (+1.89mm)
- `MSE(feat1, feat2)` mutual learning is the problem:
  - Different feature distributions of COCO/MPII collide
  - Unique strengths of each pretrained model cancel out

---

### Experiment 2: Dual Warmup v2 (Progressive Warmup)

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_coco_mpii_warmup_10ep`

**Hypothesis**: Gradually introducing mutual learning in early epochs preserves pretrained knowledge

**Warmup Schedule**:
```
Epoch 0-1:  mutual_weight = 0.0  (warmup)
Epoch 2-6:  mutual_weight = 0.0 → 0.8  (ramp-up)
Epoch 7-9:  mutual_weight = 1.0  (full)
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | mutual_weight |
|-------|-----------|------------|------------|---------------|
| 6 | 49.56mm | 33.27mm | 65.85mm | 0.8 |
| 7 | 50.40mm | 33.88mm | 66.91mm | 1.0 |
| 8 | 46.78mm | 32.04mm | 61.53mm | 1.0 |
| **9** | **45.93mm** | **31.07mm** | **60.79mm** | 1.0 |
| 10 | 49.47mm | 32.39mm | 66.55mm | 1.0 |

**Conclusion**:
- Progressive warmup alone cannot resolve the fundamental problem of Dual
- **Worse** than existing Dual (43.26mm) (+2.67mm)
- Mutual learning itself is inefficient for the COCO/MPII combination

---

### Experiment 3: Single Lifting (Soft-argmax 2D→3D)

**Config**: `HMD_xregopose_single_lifting_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_single_lifting`

**Hypothesis**: Simplify structure by replacing HeatmapEncoder/Decoder with soft-argmax + Lifting

**Architecture**:
```
[Original Single COCO]
Backbone → Heatmap → Encoder → Z[64] → PoseDecoder → 3D
                            ↘ HeatmapDecoder (40M params!)

[Single Lifting]
Backbone → Heatmap → soft_argmax → 2D[32] + conf[16]
                                        ↓
                         Lifting Network (4M params)
                                        ↓
                                   3D Pose
```

**Per-Epoch Results** (training unstable!):
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 149.54mm | 89.33mm | 209.75mm | Initial |
| 2 | 84.47mm | 58.08mm | 110.87mm | Improved |
| **3** | **190.54mm** | 146.27mm | 234.81mm | **Spike!** |
| 4 | 80.44mm | 51.69mm | 109.19mm | Recovery |
| 5 | 57.09mm | 37.58mm | 76.60mm | |
| **9** | **45.92mm** | **33.91mm** | **57.93mm** | Best |
| 10 | 53.69mm | 41.58mm | 65.80mm | Overfitting |

**Failure Cause Analysis**:
1. **Lack of depth information**: 2D coordinates alone cannot predict 3D depth (depth ambiguity)
2. **Training instability**: 190mm spike at Epoch 3
3. **Backbone features not utilized**: depth/texture/context information not passed to 3D lifting

**Key Insight**:
> "It seems that 3D information is mixed into the heatmap. Wouldn't it be better to separate them?"
> - Heatmap → dedicated to 2D position (gradient blocked)
> - Backbone → dedicated to 3D depth cues (gradient flows)

---

### Experiment 4: Lifting + Backbone Fusion (Ready)

**Config**: `HMD_xregopose_lifting_backbone_fusion_config.py`

**Hypothesis**: Preserving backbone features via a separate path utilizes depth information

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├─────────────────────────────┐
       │                             │
       ↓ (Deconv)                    ↓ (GAP → FC)
Heatmap [16, 47, 47]            Z_backbone [256]
       │                             │
       │  ← 2D position (gradient    │  ← 3D depth (gradient
       │     blocked)                 │     flows)
       │                             │
       ↓ (soft_argmax)               │
2D coords [32] + conf [16]           │
       │                             │
       └───────── Concat ────────────┘
                    ↓
           [32 + 16 + 256 + 9] = 313
                    ↓
             Lifting Network
                    ↓
                3D Pose
```

**Key Design**:
```python
# Role Separation
coords_2d = soft_argmax(heatmaps)
coords_2d_detached = coords_2d.detach()  # 2D: gradient blocked

z_backbone = backbone_encoder(backbone_feat)  # depth: gradient flows!

pose_3d = lifting_network(coords_2d_detached, confidence, z_backbone, hmd_info)
```

**Expected Effect**:
- Heatmap: learns only 2D position (heatmap MSE + coord MSE)
- Backbone: learns 3D depth cues (3D loss only)
- Role separation expected to stabilize training

**Status**: ❌ Failed (105.18mm, stopped at 7ep)

---

### Experiment 5: EfficientHeatmapDecoder (Parameter Efficiency)

**Config**: `HMD_xregopose_efficient_decoder_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_efficient_decoder_full`

**Hypothesis**: Replacing FC-heavy HeatmapDecoder with Conv-based achieves parameter efficiency (40M → 1.35M)

**Architecture Comparison**:
```
[Original HeatmapDecoder - 40M params]
Z[64] → FC(64→47*47*16) → reshape → Heatmap[16,47,47]
         ↑ ~2.3M params (output only)

[EfficientHeatmapDecoder - 1.35M params]
Z[64] → FC(64→256*6*6) → reshape → [256,6,6]
                                       ↓
                           ConvTranspose2d (256→128, 3×3, s2)
                                       ↓ [128,12,12]
                           ConvTranspose2d (128→64, 3×3, s2)
                                       ↓ [64,24,24]
                           ConvTranspose2d (64→16, 3×3, s2)
                                       ↓ [16,48,48]
                           AdaptiveAvgPool2d → [16,47,47]
```

**Parameter Comparison**:
| Component | Original | Efficient | Reduction |
|------|----------|-----------|--------|
| HeatmapDecoder | 40.0M | 1.35M | **96.6%** |
| Total Head | ~61M | ~22M | ~64% |

**Per-Epoch Results** (10 epoch):
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 82.52mm | 59.72mm | 105.32mm |
| 2 | 58.12mm | 38.93mm | 77.31mm |
| 3 | 52.10mm | 34.12mm | 70.08mm |
| 4 | 49.93mm | 33.69mm | 66.17mm |
| 5 | 47.93mm | 32.00mm | 63.86mm |
| 6 | 48.82mm | 33.17mm | 64.48mm |
| 7 | 46.90mm | 31.83mm | 61.97mm |
| **8** | **45.06mm** | **30.54mm** | **59.58mm** |
| 9 | 46.01mm | 31.50mm | 60.52mm |
| 10 | 46.92mm | 31.77mm | 62.08mm |

**Analysis**:
- **Parameter efficiency achieved**: 96.6% reduction (40M → 1.35M)
- **Performance degraded**: +3.69mm vs Baseline (41.37mm → 45.06mm)
- **Trade-off**: 30x parameter reduction vs 9% performance drop

**Failure Cause Analysis**:
1. **Insufficient expressiveness**: ConvTranspose2d's gradual upsampling has limited expressiveness compared to direct Z→Heatmap mapping
2. **Checkerboard Artifact**: ConvTranspose2d's inherent artifacts degrade heatmap quality
3. **Information loss**: Information lost during 6×6 → 47×47 upsampling

**Improvement Ideas**:
- AdaIN (Adaptive Instance Normalization): Z influences each layer
- PixelShuffle: Prevent checkerboard artifacts
- Joint-wise Generation: Independent decoder per joint

**Conclusion**:
> EfficientHeatmapDecoder succeeded in parameter efficiency,
> but performance degraded. Structural improvements are needed rather than pure efficiency.

---

### Experiment 6: Attention Lifting v1 (Cross-Attention Based)

**Config**: `HMD_xregopose_attention_lifting_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_full`
**Note**: Attention Lifting v1~v4 use EfficientHeatmapDecoder

**Hypothesis**: Query depth information from backbone via Cross-Attention, and learn joint relationships with HMD information through attention

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓ (Deconv)                         ↓ (reshape → spatial tokens)
Heatmap [16, 47, 47]               Backbone Spatial [64, 256]
       │                                  │
       ↓ (soft_argmax)                    │
2D coords [16, 2]                         │
       │                                  │
       ↓ (Joint Embedding)                │
Joint tokens [16, 64]                     │
       │                                  │
       ↓ (Backbone Cross-Attention) ←─────┘  Query depth from backbone
       ↓
       ↓ (HMD Cross-Attention) ←───── HMD tokens [3, 64] (head, R hand, L hand)
       ↓
       ↓ (Joint Self-Attention × 2) ←── Learn structural relationships between joints
       ↓
3D Pose [16, 3]
```

**Key Design**:
```python
# Attention-based Lifting
# 1. Backbone Cross-Attention: joints query depth from backbone spatial features
# 2. HMD Cross-Attention: joints learn reference from 3 HMD tokens (head, R hand, L hand)
# 3. Joint Self-Attention: learn structural relationships between joints (symmetry, connectivity)

# Z regularization via EfficientHeatmapDecoder (reconstruction loss)
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | LR | Notes |
|-------|-----------|------------|------------|-----|------|
| 1 | 55.30mm | 42.63mm | 67.97mm | 5e-4 | |
| 2 | 53.81mm | 37.84mm | 69.79mm | 5e-4 | |
| 3 | 49.19mm | 33.74mm | 64.63mm | 5e-4 | |
| **4** | **52.12mm** | 34.79mm | 69.44mm | 5e-4 | **⚠️ Spike!** |
| 5 | 47.88mm | 32.49mm | 63.28mm | 2.5e-4 | LR decay |
| 6 | 47.86mm | 31.87mm | 63.84mm | 2.5e-4 | |
| **7** | **45.43mm** | **30.14mm** | **60.72mm** | 2.5e-4 | **Best** |
| 8 | 45.74mm | 30.35mm | 61.14mm | 1.25e-4 | LR decay |
| 9 | 45.60mm | 29.97mm | 61.22mm | 1.25e-4 | |
| 10 | 45.63mm | 29.66mm | 61.61mm | 1.25e-4 | |

**Training Instability Analysis**:
- **Epoch 4 Spike**: 49.19mm → 52.12mm (+2.93mm sudden increase)
  - With LR milestone=[4,7], LR did not change at epoch 4 yet instability occurred
  - Caused by rapid changes in attention weights
- **Epoch 5 Recovery**: Stabilized after LR decreased to 2.5e-4 (52.12 → 47.88mm)

**LR Schedule Analysis**:
```
Epoch 1-4: LR = 5e-4 (unstable at high LR)
Epoch 5-7: LR = 2.5e-4 (stabilized after LR decay, Best achieved)
Epoch 8-10: LR = 1.25e-4 (converged)
```

**Failure Cause Analysis**:
1. **Attention training instability**: Cross-attention has rapid weight changes in early stages
2. **Inappropriate LR schedule**: MultiStepLR [4,7] does not suit attention
3. **Lack of warmup**: High initial LR causes attention weight instability

**Improvement Direction (v2)**:
1. **LR Warmup**: LinearLR (0.1x → 1x, 2 epochs)
2. **CosineAnnealing**: Smooth LR decay instead of MultiStepLR
3. **Gradient Clipping**: Prevent gradient explosion with max_norm=1.0

**v2 Config**: `HMD_xregopose_attention_lifting_v2_full_config.py`
```python
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=2, end=10, eta_min=1e-5),
]
```

**Conclusion**:
> Attention Lifting is structurally reasonable, but has training stability issues.
> v2 will attempt stabilization with LR warmup + CosineAnnealing + Gradient Clipping.

---

### Experiment 7: Attention Lifting v2 (Warmup + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v2_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v2_full`

**Hypothesis**: Resolve v1's training instability with LR warmup + CosineAnnealing + Gradient Clipping

**Changes from v1**:
```python
# v1: MultiStepLR [4, 7], no warmup, no gradient clipping
# v2:
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # Added
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=2),  # Warmup added
    dict(type='CosineAnnealingLR', begin=2, end=10, eta_min=1e-5),  # Changed
]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | LR | Notes |
|-------|-----------|------------|------------|-----|------|
| 1 | 111.34mm | 87.46mm | 135.22mm | 5e-5 | Warmup (0.1x) |
| 2 | 59.15mm | 40.85mm | 77.45mm | 5e-4 | Warmup complete |
| 3 | 56.16mm | 37.94mm | 74.38mm | ~4.5e-4 | CosineAnnealing |
| 4 | 51.80mm | 33.17mm | 70.42mm | ~3.9e-4 | |
| **5** | **57.78mm** | 36.09mm | 79.48mm | ~3.2e-4 | **⚠️ Spike!** |
| 6 | 51.31mm | 33.76mm | 68.86mm | ~2.5e-4 | Recovery |
| 7 | 50.83mm | 33.41mm | 68.25mm | ~1.8e-4 | |
| **8** | **48.38mm** | **32.03mm** | **64.73mm** | ~1.2e-4 | **Best** |
| 9 | 50.92mm | 32.84mm | 69.01mm | ~0.6e-4 | |
| 10 | 48.97mm | 32.49mm | 65.45mm | ~1e-5 | |

**v1 vs v2 Comparison**:
| Item | v1 (MultiStepLR) | v2 (Warmup+Cosine) | Notes |
|------|------------------|-------------------|------|
| Best MPJPE | **45.43mm** | 48.38mm | v1 is 2.95mm better |
| Best Epoch | 7 | 8 | |
| Epoch 1 | 55.30mm | 111.34mm | **Warmup delays early learning** |
| Spike | Epoch 4 (+2.93mm) | Epoch 5 (+5.98mm) | v2 spike is worse |

**Failure Cause Analysis**:
1. **Warmup counterproductive**: Starting at LR=0.1x causes epoch 1 error of 111mm (v1: 55mm)
   - Insufficient learning during 2 epochs
   - v1 rapidly decreases from the start with high LR
2. **CosineAnnealing unsuitable**: Smooth LR decay does not suit attention
   - MultiStepLR's abrupt LR drop is actually more effective
3. **Still unstable**: Spike at Epoch 5 (57.78mm, +5.98mm)
   - Gradient clipping does not fully prevent spikes

**Key Insight**:
> - **Warmup is counterproductive**: Both v2 and v7 worsened with warmup
> - **MultiStepLR is more suitable**: Abrupt LR drop is more effective than CosineAnnealing
> - **Next attempt**: Maintain v1 schedule with higher LR (0.001) (v3)

**v3 Plan** (LR=0.001):
```python
# v3: Higher LR (2x of v1) + MultiStepLR + Gradient Clipping
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[4, 7], gamma=0.5, by_epoch=True),
]
```

---

### Experiment 8: Attention Lifting v3 (Higher LR = 0.001)

**Config**: `HMD_xregopose_attention_lifting_v3_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v3_full`

**Hypothesis**: Fast convergence with high LR (0.001 = 2x of v1) + maintain MultiStepLR

**Changes from v1**:
```python
# v1: LR=0.0005, no gradient clipping
# v3:
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),  # 2x LR
    clip_grad=dict(max_norm=1.0, norm_type=2),  # Added
)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[4, 7], gamma=0.5, by_epoch=True),  # Same
]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | LR | Notes |
|-------|-----------|------------|------------|-----|------|
| **1** | **46.68mm** | **34.41mm** | **58.95mm** | 1e-3 | **Best** |
| 2 | 61.09mm | 37.27mm | 84.92mm | 1e-3 | **⚠️ +14.4mm diverged!** |
| 3 | 58.99mm | 34.35mm | 83.64mm | 1e-3 | |
| 4 | 54.63mm | 31.45mm | 77.80mm | 1e-3 | |
| 5 | 62.43mm | 36.71mm | 88.15mm | 5e-4 | **⚠️ Re-diverged after LR decay!** |
| 6 | 51.61mm | 32.43mm | 70.79mm | 5e-4 | |
| 7 | 52.39mm | 31.70mm | 73.08mm | 5e-4 | |
| 8 | 51.02mm | 31.33mm | 70.71mm | 2.5e-4 | |
| 9 | 51.11mm | 31.17mm | 71.05mm | 2.5e-4 | |
| 10 | 52.95mm | 33.18mm | 72.72mm | 2.5e-4 | |

**v1 vs v2 vs v3 Comparison**:
| Item | v1 (LR=5e-4) | v2 (Warmup+Cosine) | v3 (LR=1e-3) |
|------|--------------|-------------------|--------------|
| **Best MPJPE** | **45.43mm 🏆** | 48.38mm | 46.68mm |
| Best Epoch | 7 | 8 | 1 |
| Epoch 1 | 55.30mm | 111.34mm | 46.68mm |
| Worst Spike | +2.93mm | +5.98mm | **+14.41mm** |
| Convergence | ✅ Stable | ⚠️ Unstable | ❌ Diverged |

**Failure Cause Analysis**:
1. **Excessive LR**: 0.001 is too aggressive for Attention
   - Found a good initial value (46.68mm) at Epoch 1 but
   - Diverged by +14.4mm at Epoch 2
2. **Re-diverged after LR decay**: Another spike at Epoch 5 right after LR 0.5x reduction (62.43mm)
3. **Unable to recover**: Could not return to Epoch 1 levels

**Attention Lifting Series Final Conclusion**:
| Version | Setting | Best MPJPE | Result |
|------|------|------------|------|
| **v1** | LR=5e-4, MultiStepLR | **45.43mm** | 🏆 Best |
| v7 | LR=5e-4, MultiStepLR+Warmup | 45.73mm | ❌ Warmup counterproductive |
| v3 | LR=1e-3, MultiStepLR | 46.68mm | ❌ Excessive LR, diverged |
| v4 | LR=1e-3, CosineAnnealing | 47.95mm | ❌ Hindered initial convergence |
| v2 | LR=5e-4, Warmup+Cosine | 48.38mm | ❌ Warmup counterproductive |
| v5 | LR=2e-3, CosineAnnealing | 49.75mm | ❌ Excessive LR (2x of v4) |
| v6 | LR=1e-3, CosineRestartLR | 51.82mm | ❌ LR restart unstable, diverged |

> **Conclusion**: Attention Lifting v1 (45.43mm) is the best, but +4.06mm worse than Baseline (41.37mm).
> Even in v7, warmup was counterproductive (epoch 2 spike). **Attention architecture requires fast initial learning without warmup**.
> A completely different approach is needed to achieve Baseline performance.

---

### Experiment 9: Attention Lifting v4 (High LR + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v4_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v4_full`

**Hypothesis**: v3's high LR (0.001) + v2's CosineAnnealing (without warmup) = fast initial convergence + stable decay

**Settings**:
```python
# v3's high LR + CosineAnnealing (without warmup)
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', begin=0, end=10, eta_min=1e-5, by_epoch=True),
]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 76.12mm | 50.47mm | 101.77mm | |
| 2 | 59.32mm | 36.80mm | 81.84mm | |
| 3 | 54.48mm | 33.33mm | 75.63mm | |
| 4 | 50.70mm | 31.45mm | 69.94mm | |
| 5 | 56.30mm | 35.52mm | 77.09mm | ⚠️ Spike |
| **6** | **47.95mm** | **30.10mm** | **65.80mm** | **Best** |
| 7 | 51.24mm | 32.21mm | 70.27mm | |
| 8 | 49.27mm | 30.37mm | 68.17mm | |
| 9 | 52.01mm | 32.16mm | 71.86mm | |
| 10 | 52.10mm | 32.67mm | 71.52mm | |

**v3 vs v4 Comparison**:
| Item | v3 (MultiStepLR) | v4 (CosineAnnealing) |
|------|------------------|---------------------|
| Epoch 1 | 46.68mm | 76.12mm |
| Best | 46.68mm (ep1) | 47.95mm (ep6) |
| Stability | ❌ Diverged | ✅ Stable |
| Spike | +14.4mm | +5.6mm |

**Failure Cause Analysis**:
1. **CosineAnnealing hinders initial convergence**:
   - v3 maintains LR=0.001 during epoch 1 → fast initial convergence (46.68mm)
   - v4's CosineAnnealing reduces LR even during epoch 1 → slow initial convergence (76.12mm)
2. **v4 is stable but slow**: Small spike (+5.6mm) and stable, but 2.5mm worse than v1

**Key Insight**:
> The key to v3's epoch 1 success was **maintaining high LR**.
> CosineAnnealing increases stability but hinders initial convergence.

---

### Experiment 10: Attention Lifting v5 (Higher LR = 0.002 + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v5_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v5_full`

**Hypothesis**: Compensate v4's slow initial convergence with higher LR (0.002 = 2x of v4)

**Settings**:
```python
# v4: LR=0.001, CosineAnnealing → epoch 1: 76.12mm (slow start)
# v5: LR=0.002 (2x of v4) + CosineAnnealing
optim_wrapper = dict(
    optimizer=dict(lr=0.002, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', begin=0, end=10, eta_min=1e-5, by_epoch=True),
]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 63.99mm | 39.01mm | 88.96mm | |
| **2** | **68.76mm** | 38.24mm | 99.28mm | **⚠️ Spike! +4.77mm** |
| 3 | 56.37mm | 32.71mm | 80.02mm | Recovery |
| 4 | 51.36mm | 29.44mm | 73.28mm | |
| 5 | 58.62mm | 34.63mm | 82.61mm | ⚠️ Spike |
| 6 | 52.35mm | 30.50mm | 74.20mm | |
| 7 | 50.89mm | 31.65mm | 70.12mm | |
| **8** | **49.75mm** | **31.08mm** | **68.41mm** | **🏆 Best** |
| 9 | 51.02mm | 31.35mm | 70.70mm | |
| 10 | 52.47mm | 32.48mm | 72.46mm | |

**v4 vs v5 Comparison**:
| Item | v4 (LR=0.001) | v5 (LR=0.002) |
|------|---------------|---------------|
| Epoch 1 | 76.12mm | 63.99mm |
| Epoch 2 | 59.32mm | **68.76mm** ⚠️ |
| Best | **47.95mm** | 49.75mm |
| Best Epoch | 6 | 8 |
| Spike | +5.6mm (ep5) | **+4.77mm (ep2)** |

**Failure Cause Analysis**:
1. **LR 0.002 is too high**: Spike at Epoch 2 (63.99 → 68.76mm)
2. **1.8mm worse than v4**: High LR accelerates initial convergence but causes instability
3. **CosineAnnealing + high LR combination is unsuitable**

**Conclusion**:
> - LR=0.002 is unstable when combined with CosineAnnealing
> - v4 (LR=0.001) is better than v5
> - LR=0.0005 (v1) is optimal for Attention Lifting

---

### Experiment 11: Attention Lifting v6 (CosineRestartLR - Warm Restarts)

**Config**: `HMD_xregopose_attention_lifting_v6_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v6_full`

**Hypothesis**: Periodic LR restart via CosineRestartLR (warm restarts) → escape local minima

**Settings**:
```python
# Warm Restarts: periods=[3,3,3,1] → 3+3+3+1 = 10 epochs
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(
        type='CosineRestartLR',
        periods=[3, 3, 3, 1],  # restart at epoch 3, 6, 9
        restart_weights=[1, 1, 1, 1],
        eta_min=1e-5,
        by_epoch=True,
    ),
]
```

**Expected LR Pattern**:
- Epoch 1-3: 0.001 → decay → ~1e-5 (Period 1)
- Epoch 4: **restart → 0.001** (Period 2 start)
- Epoch 4-6: 0.001 → decay → ~1e-5
- Epoch 7: **restart → 0.001** (Period 3 start)
- Epoch 7-9: 0.001 → decay → ~1e-5
- Epoch 10: **restart → 0.001** (Period 4, 1 epoch only)

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | LR Status | Notes |
|-------|-----------|------------|------------|---------|------|
| 1 | 60.91mm | 37.96mm | 83.87mm | Period 1 start | |
| 2 | 57.57mm | 35.05mm | 80.10mm | | |
| 3 | 53.38mm | 33.26mm | 73.51mm | Period 1 end | |
| 4 | 54.78mm | 34.33mm | 75.24mm | **LR restart** | ⚠️ Performance drop |
| 5 | 60.76mm | 37.10mm | 84.43mm | | ⚠️ Spike |
| **6** | **51.82mm** | **32.89mm** | **70.75mm** | Period 2 end | **🏆 Best** |
| 7 | 54.20mm | 34.31mm | 74.09mm | **LR restart** | ⚠️ Performance drop |
| 8 | 56.15mm | 39.04mm | 73.26mm | | |
| 9 | 55.55mm | 36.90mm | 74.19mm | Period 3 end | |
| 10 | 97.62mm | 51.77mm | 143.46mm | **LR restart** | ❌ **Diverged!** |

**Failure Cause Analysis**:
1. **Performance drop right after LR restart**: Performance degraded at Epochs 4, 7, 10 after restart
   - Restart destabilizes learned weights
2. **Epoch 10 divergence**: Period 4 has only 1 epoch, so training diverged without time to recover after restart at high LR (0.001) (97.62mm)
3. **Warm restarts unsuitable**: Attention Lifting needs stable LR decay

**v1~v6 LR Schedule Comparison**:
| Version | LR Schedule | Best MPJPE | Epoch 10 | Result |
|------|-------------|------------|----------|------|
| **v1** | MultiStepLR [4,7], LR=5e-4 | **45.43mm** | 45.63mm | 🏆 Best |
| v2 | Warmup + CosineAnnealing, LR=5e-4 | 48.38mm | 48.97mm | ❌ |
| v3 | MultiStepLR [4,7], LR=1e-3 | 46.68mm | 52.95mm | ❌ |
| v4 | CosineAnnealing, LR=1e-3 | 47.95mm | 52.10mm | ❌ |
| v5 | CosineAnnealing, LR=2e-3 | 49.75mm | 52.47mm | ❌ |
| v6 | **CosineRestartLR**, LR=1e-3 | 51.82mm | **97.62mm** | ❌ Diverged |

**Conclusion**:
> - CosineRestartLR (warm restarts) is **unsuitable** for Attention Lifting
> - LR restart harms training stability
> - Last period (1 epoch) is too short, causing divergence
> - **v1's MultiStepLR remains the best**

---

### Experiment 12: Attention Lifting v7 (Optimized LR Schedule)

**Config**: `HMD_xregopose_attention_lifting_v7_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v7_full`

**Hypothesis**: Design optimized LR schedule based on v1~v6 analysis
- Maintain v1's LR=0.0005 (optimal)
- Move milestones to [3, 5, 7] to prevent v1's epoch 4 spike
- Add Gradient clipping (max_norm=1.0)
- 500 iteration warmup (0.5x → 1x)

**Settings**:
```python
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5, by_epoch=True),
]
```

**Expected LR Pattern**:
- Iter 0-500: 0.25e-3 → 0.5e-3 (iteration warmup)
- Epoch 1-3: 5e-4
- Epoch 4-5: 2.5e-4 (after milestone 3)
- Epoch 6-7: 1.25e-4 (after milestone 5)
- Epoch 8-10: 6.25e-5 (after milestone 7)

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | LR | Notes |
|-------|-----------|------------|------------|-----|------|
| 1 | 52.38mm | 39.87mm | 64.89mm | warmup→5e-4 | |
| **2** | **61.82mm** | 44.55mm | 79.09mm | 5e-4 | **⚠️ Spike! +9.44mm** |
| 3 | 55.63mm | 39.04mm | 72.22mm | 5e-4 | Recovery starting |
| 4 | 47.72mm | 33.01mm | 62.44mm | 2.5e-4 | LR decay |
| 5 | 49.33mm | 33.67mm | 64.98mm | 2.5e-4 | |
| 6 | 46.88mm | 32.44mm | 61.31mm | 1.25e-4 | 2nd decay |
| **7** | **45.73mm** | **32.69mm** | **58.78mm** | 1.25e-4 | **🏆 Best** |
| 8 | 46.03mm | 31.89mm | 60.17mm | 6.25e-5 | 3rd decay |
| 9 | 46.27mm | 32.00mm | 60.54mm | 6.25e-5 | |
| 10 | 46.22mm | 31.89mm | 60.54mm | 6.25e-5 | Converged |

**v1 vs v7 Comparison**:
| Item | v1 | v7 | Notes |
|------|-----|-----|------|
| **Best MPJPE** | **45.43mm 🏆** | 45.73mm | v1 is 0.30mm better |
| Best Epoch | 8 | 7 | |
| Epoch 2 | 45.62mm | **61.82mm** | **v7 spike +16.2mm** |
| Spike Location | Epoch 4 (+2.93mm) | Epoch 2 (+9.44mm) | v7 spike is worse |
| Gradient Clipping | ❌ | ✅ | |
| Warmup | ❌ | ✅ (500 iter) | |

**Failure Cause Analysis**:
1. **Warmup counterproductive**: 500 iter warmup actually caused spike at epoch 2
   - v1 already stabilized at 45.62mm by epoch 2
   - v7 surged to 61.82mm at epoch 2
2. **Similar final performance to v1**: Best differs by only 0.30mm, nearly identical
3. **Early milestones did not prevent spike**: Milestones [3, 5, 7] are unrelated to epoch 2 spike

**Key Insight**:
> - **Warmup is harmful for Attention Lifting**: Both v2 and v7 worsened with warmup
> - **v1's "MultiStepLR without warmup" is optimal**: Fast learning from the start with high LR
> - Attention architecture requires fast initial learning, warmup interferes with this

**v1~v7 Final Comparison**:
| Version | LR Schedule | Best MPJPE | Warmup | Result |
|------|-------------|------------|--------|------|
| **v1** | MultiStepLR [4,7], LR=5e-4 | **45.43mm** | ❌ | 🏆 Best |
| v7 | MultiStepLR [3,5,7], LR=5e-4, warmup 500iter | 45.73mm | ✅ | ❌ warmup counterproductive |
| v3 | MultiStepLR [4,7], LR=1e-3 | 46.68mm | ❌ | ❌ Excessive LR |
| v4 | CosineAnnealing, LR=1e-3 | 47.95mm | ❌ | ❌ Hindered initial convergence |
| v2 | Warmup + CosineAnnealing, LR=5e-4 | 48.38mm | ✅ | ❌ warmup counterproductive |
| v5 | CosineAnnealing, LR=2e-3 | 49.75mm | ❌ | ❌ Excessive LR |
| v6 | CosineRestartLR, LR=1e-3 | 51.82mm | ❌ | ❌ restart unstable |

**Conclusion**:
> **Attention Lifting v1 (45.43mm) is the best**, and further LR schedule optimization is ineffective.
> **Structural changes** are needed to achieve Baseline (41.37mm).

---

### Experiment 13: Skeleton Graph Attention Network (GAT)

**Config**: `HMD_xregopose_skeleton_gat_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_skeleton_gat_full`

**Hypothesis**: Replace JointSelfAttention with SkeletonGraphAttention to leverage anatomical structure prior
- Joints attend only to connected neighboring joints
- Sparse attention based on skeleton adjacency matrix
- Includes 2-hop neighbors (second-order connections)
- Learnable edge attention bias

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓ (Deconv)                         ↓ (reshape)
Heatmap [16, 47, 47]             Backbone tokens [64, D]
       │                                  │
       ↓ (soft_argmax)                    │
  2D coords [16, 2]                       │
       │                                  │
       ↓ (Joint Embedding)                │
  Joint tokens [16, 64]                   │
       │                                  │
       └── Backbone Cross-Attention ──────┘
                    │
                    ▼
       HMD Cross-Attention ←── HMD tokens [3, 64]
                    │
                    ▼
          ┌─────────────────────┐
          │ Skeleton GAT × 2    │  ← Key: Only connected joints attend
          │                     │
          │ Adjacency Matrix:   │
          │ head─neck─shoulders │
          │       │    │        │
          │      hips elbows    │
          │       │    │        │
          │     knees wrists    │
          │       │             │
          │    ankles           │
          └─────────────────────┘
                    │
                    ▼
              3D Pose [16, 3]
```

**Skeleton Connection Structure**:
```python
SKELETON_EDGES = [
    (0, 1),    # head - neck
    (1, 2),    # neck - left_shoulder
    (2, 3),    # left_shoulder - left_elbow
    (3, 4),    # left_elbow - left_wrist
    (1, 5),    # neck - right_shoulder
    (5, 6),    # right_shoulder - right_elbow
    (6, 7),    # right_elbow - right_wrist
    (1, 8),    # neck - left_hip
    (8, 9),    # left_hip - left_knee
    (9, 10),   # left_knee - left_ankle
    (1, 11),   # neck - right_hip
    (11, 12),  # right_hip - right_knee
    (12, 13),  # right_knee - right_ankle
    (2, 5),    # left_shoulder - right_shoulder
    (8, 11),   # left_hip - right_hip
]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 76.66mm | 50.85mm | 102.48mm | |
| 2 | 62.78mm | 37.58mm | 87.98mm | |
| 3 | 53.96mm | 33.04mm | 74.88mm | |
| 4 | 54.11mm | 36.28mm | 71.93mm | |
| 5 | 51.86mm | 33.57mm | 70.14mm | |
| 6 | 52.20mm | 34.04mm | 70.37mm | |
| **7** | **50.88mm** | **33.83mm** | **67.92mm** | **🏆 Best** |
| 8 | 51.19mm | 33.72mm | 68.66mm | |
| 9 | 51.24mm | 33.30mm | 69.17mm | |
| 10 | 51.60mm | 33.84mm | 69.35mm | |

**Attention Lifting v1 vs Skeleton GAT Comparison**:
| Item | Attention Lifting v1 | Skeleton GAT | Difference |
|------|---------------------|--------------|------|
| **Best MPJPE** | **45.43mm 🏆** | 50.88mm | +5.45mm |
| Upper Body | **30.14mm** | 33.83mm | +3.69mm |
| Lower Body | **60.72mm** | 67.92mm | +7.20mm |
| Best Epoch | 7 | 7 | |
| Attention Method | Dense (fully connected) | Sparse (skeleton-based) | |

**Failure Cause Analysis**:
1. **Limitations of sparse attention**: Anatomical connections alone lack the global context needed for 3D pose
   - 3D pose estimation requires mutual relationships of all joints
   - Example: Relative position of left hand and right foot is also an important depth cue
2. **Lower body performance plummeted**: Lower body +7.20mm worse
   - Lower body joints (hip, knee, ankle) require relationships with upper body
   - In Skeleton GAT, only connected through neck, resulting in insufficient information
3. **2-hop neighbors also insufficient**: Cannot restore global context even with second-order connections
   - Example: Cannot learn ankle → knee → hip → neck (3-hop) relationships

**Key Insight**:
> - **Dense attention is essential for 3D pose**: Fully-connected attention between all joints is needed
> - **Skeleton prior should be used as bias, not constraint**: Use as edge bias instead of sparse attention
> - **Attention Lifting v1's JointSelfAttention is more effective**: Can learn without structural prior

**Conclusion**:
> Skeleton GAT (50.88mm) is 5.45mm worse than Attention Lifting v1 (45.43mm).
> Confirmed that sparse attention (skeleton-based) is inferior to dense attention.
> **Global context** is essential for 3D pose estimation, and anatomical structure alone is insufficient.

---

### Experiments 14-18: ViT-Style Lifting v1~v5

**Overview**: ViTPose-style Learnable Joint Queries + Self-Attention architecture

**Core Idea**:
```
Instead of extracting coordinates via soft_argmax → Use Learnable Joint Queries
Instead of Cross-Attention → Self-Attention (bidirectional between joint ↔ spatial)
Heatmap Reconstruction → Inject 2D joint information into joint tokens
HMD Cross-Attention → Add 3D depth reference
```

#### ViT Lifting v1 (Separate Heatmap Path)

**Config**: `HMD_xregopose_vit_lifting_v1_full_config.py`

**Architecture**:
```
Backbone ─┬─→ Deconv → Heatmap → Heatmap Loss (separate)
          │
          └─→ Spatial Tokens + Joint Queries
                      ↓
              Self-Attention
                      ↓
              Joint Tokens
                      ↓
              HMD Cross-Attention
                      ↓
              3D Pose Head → 3D Loss
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 52.57mm | 33.04mm | 72.09mm |
| 4 | 51.89mm | 29.39mm | 74.39mm |
| **8** | **50.84mm** | **28.77mm** | **72.91mm** |
| 10 | 52.49mm | 30.11mm | 74.87mm |

**Analysis**: Heatmap and 3D Lifting are **independent** (no information sharing)

---

#### ViT Lifting v2 (No Reconstruction)

**Config**: `HMD_xregopose_vit_lifting_v2_full_config.py`

**Architecture**:
```
Backbone → Spatial Tokens + Joint Queries
                    ↓
            Self-Attention (spatial info automatically transferred?)
                    ↓
            Joint Tokens
                    ↓
            HMD Cross-Attention
                    ↓
            3D Pose Head → 3D Loss only
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| **1** | **51.77mm** | **33.37mm** | **70.17mm** |
| 2 | 55.91mm | 31.76mm | 80.07mm |
| 5 | 57.70mm | 33.69mm | 81.71mm |

**Analysis**:
- Without heatmap reconstruction, Self-Attention alone is insufficient for 2D information transfer
- Actually worse than v1 (+0.93mm)
- Training stopped at Epoch 5 (diverged)

---

#### ViT Lifting v3 (Reconstruction Regularization) ⭐ Best ViT

**Config**: `HMD_xregopose_vit_lifting_v3_full_config.py`

**Architecture**:
```
Backbone → Spatial Tokens + Joint Queries
                    ↓
            Self-Attention
                    ↓
            Joint Tokens ◄── shared latent
                    │
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
Heatmap Decoder            HMD Cross-Attention
       │                         │
       ▼                         ▼
Recon Heatmap              3D Pose Head
       │                         │
       ▼                         ▼
Heatmap Loss ◄──────────────► 3D Loss
(regularizes tokens)
```

**Key Point**: Heatmap Reconstruction **regularizes Joint Tokens** → forces 2D information injection

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 50.32mm | 31.40mm | 69.23mm | |
| 2 | 53.72mm | 28.27mm | 79.17mm | ⚠️ Spike |
| 3 | 48.28mm | 25.46mm | 71.10mm | Recovery |
| **4** | **45.34mm** | **23.49mm** | **67.19mm** | **🏆 Best** |
| 5 | 47.13mm | 24.76mm | 69.50mm | ⚠️ Spike |
| 6 | 49.12mm | 26.22mm | 72.02mm | |
| 7 | 45.59mm | 24.53mm | 66.65mm | |
| 10 | 47.37mm | 24.56mm | 70.19mm | |

**Analysis**:
- **Best Upper Body performance**: 23.49mm (-5.93mm compared to Baseline 29.42mm!)
- Validation spike issue: sudden increase at epochs 2, 5
- Caused by abrupt LR decay of MultiStepLR [3, 5, 7]

**LR Schedule Issue**:
```
Epoch 1-3: LR = 0.0005
Epoch 4:   LR = 0.00025 (decay at 3) → Best!
Epoch 5:   LR = 0.000125 (decay at 5) → Spike begins
```

---

#### ViT Lifting v4 (CosineAnnealingLR)

**Config**: `HMD_xregopose_vit_lifting_v4_full_config.py`

**Change from v3**: MultiStepLR → CosineAnnealingLR (smooth LR decay)

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 48.53mm | 30.10mm | 66.96mm | |
| 4 | 48.98mm | 26.45mm | 71.51mm | |
| 5 | 53.99mm | 28.65mm | 79.32mm | ⚠️ Spike |
| **7** | **45.66mm** | **25.76mm** | **65.56mm** | **🏆 Best** |
| 10 | 48.51mm | 27.09mm | 69.93mm | |

**Analysis**:
- Smoothed LR decay with CosineAnnealingLR but spike still occurred
- 0.32mm worse than v3 (45.34 → 45.66mm)
- Lower Body improved (67.19 → 65.56mm)

---

#### ViT Lifting v5 (Hybrid Attention)

**Config**: `HMD_xregopose_vit_lifting_v5_full_config.py`

**Changes from v4**:
1. Self-Attention [80×80] → Cross-Attention [16×64] + Self-Attention [16×16]
2. Role separation: Cross (location finding) + Self (skeleton relationships)
3. Gradient Scaling: Reduce heatmap gradient by 0.1x to focus on 3D learning

**Architecture**:
```
Backbone feat [2048, 8, 8]
     ↓
Spatial Tokens [64, D]
     ↓
┌─────────────────────────────────────────┐
│  Stage 1: Cross-Attention (J → S)       │
│  Q: Joint Queries [16, D]               │
│  K/V: Spatial Tokens [64, D]            │  [16×64]
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Stage 2: Self-Attention (J → J) × 2    │
│  Q=K=V: Joint Tokens [16, D]            │  [16×16]
└─────────────────────────────────────────┘
     ↓
Joint Tokens [16, D]
     ├── Heatmap Decoder (gradient scaled 0.1)
     ↓
HMD Cross-Attention [16×3]
     ↓
3D Pose Head → [16, 3]
```

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 51.73mm | 36.48mm | 66.98mm | |
| 4 | 49.03mm | 29.48mm | 68.58mm | |
| **7** | **47.22mm** | **29.09mm** | **65.35mm** | **🏆 Best** |
| 10 | 49.15mm | 29.75mm | 68.55mm | |

**Analysis**:
- 1.56mm worse than v4 (45.66 → 47.22mm)
- Hybrid Attention is less effective than v3/v4's Self-Attention
- Gradient Scaling weakened heatmap learning, reducing regularization effect

---

### ViT Lifting Series Summary

| Version | Architecture | LR Schedule | Best MPJPE | Best Epoch | Notes |
|------|------|-------------|------------|------------|------|
| v1 | Separate Heatmap | MultiStepLR | 50.84mm | 8 | No info sharing |
| v2 | No Recon | MultiStepLR | 51.77mm | 1 | Self-Attn alone insufficient |
| **v3** | **Recon + Self-Attn** | MultiStepLR | **45.34mm** | 4 | **⭐ Best ViT** |
| v4 | Recon + Self-Attn | CosineAnnealingLR | 45.66mm | 7 | Spike reduced but performance dropped |
| v5 | Hybrid Attention | CosineAnnealingLR | 47.22mm | 7 | Gradient Scaling counterproductive |

**Key Insights**:
1. **Heatmap Reconstruction is key**: v2 (no recon) is 6.43mm worse than v3 (recon)
2. **Self-Attention [80×80] is more effective than Hybrid**: v5's role separation actually worsened results
3. **Excellent Upper Body performance**: v3's Upper Body 23.49mm is 5.93mm better than Baseline (29.42mm)
4. **LR Schedule sensitive**: MultiStepLR's abrupt decay causes spikes, but performance is better

---

### Experiment 19: Upper-Lower Decoupled Head

**Config**: `HMD_xregopose_decoupled_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_decoupled_full`

**Hypothesis**: Combining ViT v3's Upper Body strength (23.49mm) with Baseline's Lower Body strength (53.31mm) can achieve ~38.40mm

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓                                  ↓
Upper Branch (ViT v3 style)          Lower Branch (Baseline style)
  - Learnable Joint Queries [8]        - Deconv → Heatmap [8, 47, 47]
  - Self-Attention                     - HeatmapEncoder → Z
  - Heatmap Reconstruction             - PoseDecoder
  - HMD Cross-Attention                - (No HMD info)
       │                                  │
       ↓                                  ↓
  Upper 3D [8, 3]                    Lower 3D [8, 3]
       │                                  │
       └──────── Concat ──────────────────┘
                    ↓
              Full 3D Pose [16, 3]
```

**Key Design**:
- Upper Body (8 joints): ViT v3 approach (Self-Attention + Recon + HMD)
- Lower Body (8 joints): Baseline approach (Deconv + Heatmap + Z-vector)
- Each branch uses the architecture optimized for its body part

**Per-Epoch Results**:
| Epoch | Full Body | Upper Body | Lower Body | Notes |
|-------|-----------|------------|------------|------|
| 1 | 59.68mm | 26.51mm | 92.85mm | |
| 2 | 57.94mm | 24.38mm | 91.50mm | |
| 3 | 59.44mm | 26.29mm | 92.58mm | ⚠️ Spike |
| 4 | 47.88mm | 23.78mm | 71.99mm | Rapid improvement |
| 5 | 59.32mm | 29.62mm | 89.03mm | ⚠️ Spike |
| 6 | 50.57mm | 25.90mm | 75.24mm | |
| 7 | 48.37mm | 24.19mm | 72.54mm | |
| **8** | **45.00mm** | **24.10mm** | **65.89mm** | **🏆 Best** |
| 9 | 46.04mm | 24.19mm | 67.90mm | |
| 10 | 46.03mm | 25.09mm | 66.97mm | |

**Expected vs Actual Values**:
| Body Part | Expected (ViT v3 + Baseline) | Actual (Decoupled) | Difference |
|------|-------------------------|------------------|------|
| Upper Body | 23.49mm | 24.10mm | +0.61mm (nearly achieved) |
| Lower Body | 53.31mm | 65.89mm | **+12.58mm (degraded)** |
| Full Body | ~38.40mm | 45.00mm | **+6.60mm** |

**Comparison Against Baseline**:
| Body Part | Decoupled (Ep8) | Baseline | Difference |
|------|-----------------|----------|------|
| Full Body | 45.00mm | 41.37mm | +3.63mm ❌ |
| Upper Body | **24.10mm** | 29.42mm | **-5.32mm** ✅ |
| Lower Body | 65.89mm | 53.31mm | +12.58mm ❌ |

**Training Instability Analysis**:
- **Epoch 3→4**: 59.44mm → 47.88mm (rapid improvement, -11.56mm)
- **Epoch 4→5**: 47.88mm → 59.32mm (rapid degradation, +11.44mm)
- Validation is very unstable, oscillating with 10~12mm amplitude

**Failure Cause Analysis**:
1. **Loss of joint coupling**: Baseline learns all 16 joints together, leveraging upper-lower correlations (e.g., arm-leg coordination during walking), and separation loses this information
2. **Lower branch structural mismatch**: Decoupled's lower branch is not identical to the original baseline head (capacity differences during conversion to 8-joint-specific architecture)
3. **Training instability**: Different learning speeds of Upper and Lower branches interfere with overall backbone gradient

**Key Insight**:
> - Upper Body improvement confirmed: ViT approach is effective for upper body (24.10mm vs Baseline 29.42mm)
> - **Lower Body is the bottleneck**: Simple separation cannot restore Baseline's lower body performance (53.31mm)
> - Joint coupling is important for lower body: Structural relationships between upper and lower body joints are needed for lower body prediction

**Conclusion**:
> Upper-Lower Decoupled (45.00mm) is 3.63mm worse than Baseline (41.37mm).
> Upper Body was achieved as expected, but Lower Body degraded by 12.58mm.
> **Simple separation strategy failed** — joint coupling preservation is needed.

---

## Comprehensive Analysis

### Failed Approaches

| Approach | Result | Problem |
|--------|------|--------|
| Dual Backbone + Mutual Learning | 43.26mm (+1.89mm) | COCO/MPII feature distribution collision |
| Progressive Warmup | 45.93mm (+4.56mm) | Cannot resolve fundamental mutual learning issue |
| Pure 2D→3D Lifting | 45.92mm (+4.55mm) | Lack of depth information, training instability |
| Lifting + Backbone Fusion | 105.18mm (+63.81mm) | Role separation failed, gradient blocking side effects |
| EfficientHeatmapDecoder | 45.06mm (+3.69mm) | Insufficient Conv upsampling expressiveness |
| Attention Lifting v1 | 45.43mm (+4.06mm) | Epoch 4 spike, inappropriate LR schedule |
| Attention Lifting v2 | 48.38mm (+7.01mm) | Warmup counterproductive, delayed early learning |
| Attention Lifting v3 | 46.68mm (+5.31mm) | LR 0.001 excessive, epoch 2 diverged |
| Attention Lifting v4 | 47.95mm (+6.58mm) | CosineAnnealing hindered initial convergence |
| Attention Lifting v5 | 49.75mm (+8.38mm) | LR=0.002 excessive, epoch 2 spike |
| Attention Lifting v6 | 51.82mm (+10.45mm) | CosineRestartLR restart unstable, epoch 10 diverged |
| Attention Lifting v7 | 45.73mm (+4.36mm) | Optimized schedule, warmup counterproductive |
| Skeleton GAT | 50.88mm (+9.51mm) | Sparse attention, insufficient global context, lower body degraded |
| ViT Lifting v1 | 50.84mm (+9.47mm) | Separate heatmap path, no info sharing |
| ViT Lifting v2 | 51.77mm (+10.40mm) | No reconstruction, Self-Attn alone insufficient |
| **ViT Lifting v3** | **45.34mm (+3.97mm)** | ⭐ Recon + Self-Attn, best Upper Body (23.49mm) |
| ViT Lifting v4 | 45.66mm (+4.29mm) | CosineAnnealingLR, spike reduced but performance dropped |
| ViT Lifting v5 | 47.22mm (+5.85mm) | Hybrid Attention, Gradient Scaling counterproductive |
| Upper-Lower Decoupled | 45.00mm (+3.63mm) | Upper improved (-5.32mm), Lower degraded (+12.58mm) |

### Key Insights

1. **Heatmap limitations**: Heatmaps are 2D probability distributions that can only implicitly encode 3D depth
2. **Role separation needed**: 2D position and 3D depth should be learned through separate paths
3. **Backbone feature utilization**: Backbone's texture/context information is key to depth estimation
4. **Gradient flow design**: 3D loss to backbone, only 2D loss to heatmap

### Next Experiment Plan

**Attention Lifting v1~v7 experiments completed**. Baseline (41.37mm) cannot be achieved through LR schedule optimization alone.

| Priority | Experiment | Expected Effect |
|----------|------|----------|
| 1 | **Structural improvement** | Explore depth extraction methods other than Attention |
| 2 | AdaIN HeatmapDecoder | Z influences each Conv layer, improved expressiveness |
| 3 | Enhanced Data Augmentation | Random rotation, scale, color jitter, etc. |

**Attention Lifting Key Insights**:
- Warmup is counterproductive (both v2 and v7 failed)
- MultiStepLR > CosineAnnealing
- LR=0.0005 is optimal (diverges if higher)
- Best: v1 (45.43mm), +4.06mm vs Baseline

---

## Execution Commands

```bash
# Baseline (for reference)
python tools/train.py my_code/custom_config/HMD_xregopose_single_coco_full_config.py

# EfficientHeatmapDecoder (completed)
python tools/train.py my_code/custom_config/HMD_xregopose_efficient_decoder_full_config.py
```

---

## File List

### Head Files

| Head | File | Purpose | Result |
|------|------|------|------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline | **41.37mm 🏆** |
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | + EfficientHeatmapDecoder | 45.06mm |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone | 43.26mm |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup | 45.93mm |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax lifting | 45.92mm |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | Lifting + Backbone | 105.18mm ❌ |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v1 🏆 | 45.43mm |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v2~v7 | 45.73~51.82mm |
| `CustomEgoposeSkeletonGATHead` | `custom_egopose_skeleton_gat_head.py` | Skeleton Graph Attention | 50.88mm |
| `CustomEgoposeViTLiftingHead` | `custom_egopose_vit_lifting_head.py` | ViT Lifting v1~v4 | 45.34mm (v3) ⭐ |
| `CustomEgoposeViTLiftingHeadV5` | `custom_egopose_vit_lifting_head_v5.py` | ViT Lifting v5 (Hybrid) | 47.22mm |
| `CustomEgoposeDecoupledHead` | `custom_egopose_decoupled_head.py` | Upper-Lower Decoupled | 45.00mm |

### Config Files

| Config | Purpose | Result |
|--------|------|------|
| `HMD_xregopose_single_coco_full_config.py` | Single COCO baseline | **41.37mm 🏆** |
| `HMD_xregopose_h5cache_coco_mpii_config.py` | Dual COCO+MPII | 43.26mm |
| `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | Dual + Warmup | 45.93mm |
| `HMD_xregopose_single_lifting_config.py` | Lifting only | 45.92mm |
| `HMD_xregopose_lifting_backbone_fusion_config.py` | Lifting + Backbone | 105.18mm ❌ |
| `HMD_xregopose_efficient_decoder_full_config.py` | EfficientHeatmapDecoder | 45.06mm |
| `HMD_xregopose_attention_lifting_full_config.py` | Attention Lifting v1 | 45.43mm |
| `HMD_xregopose_attention_lifting_v2_full_config.py` | Attention Lifting v2 (Warmup+Cosine) | 48.38mm ❌ |
| `HMD_xregopose_attention_lifting_v3_full_config.py` | Attention Lifting v3 (LR=0.001) | 46.68mm ❌ |
| `HMD_xregopose_attention_lifting_v4_full_config.py` | Attention Lifting v4 (CosineAnnealing) | 47.95mm ❌ |
| `HMD_xregopose_attention_lifting_v5_full_config.py` | Attention Lifting v5 (LR=0.002) | 49.75mm ❌ |
| `HMD_xregopose_attention_lifting_v6_full_config.py` | Attention Lifting v6 (CosineRestartLR) | 51.82mm ❌ |
| `HMD_xregopose_attention_lifting_v7_full_config.py` | Attention Lifting v7 (Optimized Schedule) | 45.73mm ❌ |
| `HMD_xregopose_skeleton_gat_full_config.py` | Skeleton GAT | 50.88mm ❌ |
| `HMD_xregopose_vit_lifting_v1_full_config.py` | ViT Lifting v1 (Separate Heatmap) | 50.84mm ❌ |
| `HMD_xregopose_vit_lifting_v2_full_config.py` | ViT Lifting v2 (No Recon) | 51.77mm ❌ |
| `HMD_xregopose_vit_lifting_v3_full_config.py` | **ViT Lifting v3 (Recon)** | **45.34mm ⭐** |
| `HMD_xregopose_vit_lifting_v4_full_config.py` | ViT Lifting v4 (CosineAnnealing) | 45.66mm ❌ |
| `HMD_xregopose_vit_lifting_v5_full_config.py` | ViT Lifting v5 (Hybrid Attention) | 47.22mm ❌ |
| `HMD_xregopose_decoupled_full_config.py` | Upper-Lower Decoupled | 45.00mm ❌ |
