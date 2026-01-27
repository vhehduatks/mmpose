# Cascaded Pose Refinement with Kinematic Prior

> Created: 2026-01-27

## Motivation

From 20 experiments documented in EXPERIMENT_RESULTS.md, the Baseline (41.37mm) remains the only architecture that achieves good lower body performance (53.31mm). Every alternative architecture degrades lower body by 3–20mm.

**Lower Body Performance Ranking** (selected):
| Model | Lower Body | vs Baseline |
|-------|------------|-------------|
| Baseline | 53.31mm | — |
| Dual COCO+MPII | 56.48mm | +3.17mm |
| Single Lifting | 57.93mm | +4.62mm |
| Attention v1 | 60.72mm | +7.41mm |
| Decoupled | 65.89mm | +12.58mm |
| ViT v3 | 67.19mm | +13.88mm |
| ViT v1 | 72.91mm | +19.60mm |

**Key insight**: Don't replace the Baseline — build on top of it.

## Problem Analysis

### Why Baseline is best for lower body
1. **Unified Z[64]**: All 16 joints encoded into a single latent vector. PoseDecoder predicts all joints simultaneously → upper-lower correlations preserved
2. **MLP-based prediction**: No attention instability. BatchNorm + Dropout regularize well for 210K samples
3. **HMD as auxiliary**: HMD info (9→36) is concatenated to Z as a small fraction, not dominating prediction

### Why alternatives fail
1. **Attention mechanisms** (v1–v7, GAT, ViT): Training instability, HMD biases toward upper body
2. **Separate branches** (Decoupled): Loses joint coupling (+12.58mm lower body degradation)
3. **Lifting approaches**: Insufficient depth information for lower body

### Why lower body is fundamentally harder
- No HMD anchor (only head + 2 hands available)
- Distant from camera → smaller in image, less texture detail
- Higher depth ambiguity (multiple valid 3D configurations for same 2D)
- Lower body error is 1.81× upper body in Baseline (53.31 vs 29.42mm)

## Differentiation from Queued/Existing Approaches

| Approach | Strategy | Problem |
|----------|----------|---------|
| Attention Z Encoder (queued) | Replace GAP with cross-attention | Single-stage, adds attention (instability risk) |
| ViT v6 Lower Body (queued) | ViT + loss weighting | Entire pipeline is attention-based |
| ViT v6 SPT+LSA (queued) | Small dataset optimization | Still ViT architecture |
| Upper-Lower Decoupled (exp 19) | Separate branches | Loses joint coupling |
| Hierarchical (implemented) | Upper→Lower with attention | Attention-based, error propagation |
| **This proposal** | **Baseline + MLP refinement** | **Preserves Baseline, no attention, kinematic prior** |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: Standard Baseline (unchanged, proven at 41.37mm)       │
│                                                                   │
│  Backbone feat [2048, 8, 8]                                       │
│       ↓                                                           │
│  Deconv → Heatmap [16, 47, 47]                                   │
│       ↓                                                           │
│  Encoder(GAP) + HMD → Z[64]                                      │
│       ↓                                                           │
│  PoseDecoder → Coarse 3D Pose [16, 3]                            │
│       ↓                                                           │
│  HeatmapDecoder → Recon Heatmap (regularization)                 │
│                                                                   │
│  Losses: pose_l2norm, cosine_similarity, limb_length,            │
│          heatmap_recon, hmd_recon, kpt_2d                        │
└──────────────────────────────────────────────────────────────────┘
                    │  Coarse Pose [16, 3]
                    │  Z[64]
                    │  Backbone feat [2048, 8, 8]
                    │  Heatmap [16, 47, 47]
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: Kinematic-Aware Pose Refinement (new, lightweight)     │
│                                                                   │
│  Step 1: Per-Joint Spatial Features                               │
│    soft_argmax(heatmap) → 2D coords [16, 2] (detached)          │
│    grid_sample(backbone_feat, coords) → [16, 2048]              │
│    Linear(2048→64) → spatial_feat [16, 64]                       │
│                                                                   │
│  Step 2: Pose Context Encoding                                    │
│    coarse_pose [16, 3] → flatten → Linear(48→128) → [128]       │
│                                                                   │
│  Step 3: Kinematic Chain Features                                 │
│    15 bones: bone_vec[3] + bone_len[1] = [60]                   │
│    Linear(60→64) → kin_feat [64]                                 │
│                                                                   │
│  Step 4: Per-Joint Refinement                                     │
│    For each joint j:                                              │
│      concat(pose_j[3], spatial[64], Z[64], pose_ctx[128],        │
│             kin_feat[64]) = [323]                                 │
│    Shared MLP: 323→256 (BN+ReLU+Dropout, residual) → 3          │
│    Δpose = MLP output                                            │
│                                                                   │
│  Refined Pose = Coarse Pose + Δpose (residual)                   │
│                                                                   │
│  Losses: pose_l2norm_refined, bone_length, symmetry              │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Per-Joint Spatial Features via Grid Sampling
- GAP aggregates the entire 8×8 feature map, losing WHERE each joint is
- Grid sampling extracts backbone features AT each joint's 2D location
- Lower body joints get features from their specific image region
- Coords are detached to protect heatmap learning

### 2. Kinematic Chain Features
- 15 bone vectors (direction + length) give explicit geometric structure
- Enables learning: "if thigh points downward with length 0.45m, knee is ~0.45m below hip"
- Top-down propagation: well-predicted upper joints help lower body
- Kinematic tree:
```
Spine2 (near head, well-predicted)
  ├→ LeftArm → LeftForeArm → LeftHand
  ├→ RightArm → RightForeArm → RightHand
  ├→ LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase
  └→ RightUpLeg → RightLeg → RightFoot → RightToeBase
```

### 3. Residual Refinement (Δpose)
- Predicts only the correction, not the absolute pose
- Stage 1 handles the bulk (41.37mm overall), Stage 2 learns the error pattern
- Much easier to learn a correction than an absolute prediction

### 4. MLP-Based (No Attention)
- Same building blocks as Baseline (Linear + BN + Dropout)
- No training instability, no spikes
- Works well with 210K samples

### 5. Joint Coupling Preserved
- `pose_feat` (all 48 dims → 128) and `kin_feat` (all 15 bones → 64) broadcast to every joint
- Every joint sees full-body context during refinement

## Loss Function Design

### Stage 1 (identical to Baseline)
| Loss | Target | Weight |
|------|--------|--------|
| loss_kpt | Heatmap MSE | 1000 |
| loss_pose_l2norm | Coarse 3D Pose L2 | 1.0 |
| loss_cosine_similarity | Limb direction | 0.1 |
| loss_limb_length | L1 distance | 0.25 |
| loss_heatmap_recon | Recon heatmap MSE | 500 |
| loss_hmd | HMD reconstruction | 1.0 |

### Stage 2 (refinement)
| Loss | Target | Weight |
|------|--------|--------|
| loss_pose_l2norm_refined | Refined 3D Pose L2 | 1.0 |
| loss_bone_length | Per-limb bone length | 0.5 |
| loss_symmetry | L-R limb symmetry | 0.1 |

### Training Strategy
- End-to-end (not frozen Stage 1)
- Intermediate supervision on coarse pose prevents Stage 1 degradation
- Same LR schedule as Baseline: MultiStepLR [4,7], gamma=0.5, LR=0.0005
- 2D coords for grid sampling are detached (protect heatmap gradients)

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| Stage 1 (Baseline, unchanged) | ~61M |
| spatial_proj: Linear(2048→64) | 131K |
| pose_encoder: Linear(48→128) | 6K |
| kin_encoder: Linear(60→64) | 4K |
| RefinementMLP (1 residual block) | ~250K |
| **Stage 2 Total** | **~400K (<1% of total)** |

## Expected Effect

```
Baseline (reference):
  Upper: 29.42mm    Lower: 53.31mm    Full: 41.37mm

Expected refinement gains:
  Per-joint spatial features: -3 to -5mm lower body
  Kinematic chain features:  -2 to -3mm lower body
  Residual correction:       -1 to -2mm overall

Conservative estimate:
  Upper: ~28mm    Lower: ~46mm    Full: ~37mm

Optimistic estimate:
  Upper: ~27mm    Lower: ~43mm    Full: ~35mm
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Stage 2 learns identity (Δ≈0) | Intermediate supervision on coarse; refinement-specific losses push corrections |
| Grid sampling gradients unstable | 2D coords detached from heatmap |
| Overfitting Stage 2 on 210K | Only ~400K params, MLP with BN+Dropout |
| Stage 1 degrades during training | Intermediate loss on coarse pose; worst case freeze Stage 1 |

## Implementation Files

| File | Purpose |
|------|---------|
| `custom_egopose_cascaded_refinement_head.py` | Head implementation |
| `HMD_xregopose_cascaded_refinement_small_config.py` | Smoke test config |
| `HMD_xregopose_cascaded_refinement_full_config.py` | Full training config |

## Skeleton Indices (xRegopose)

```
0: Spine2, 1: Head, 2: LeftArm, 3: LeftForeArm, 4: LeftHand,
5: RightArm, 6: RightForeArm, 7: RightHand, 8: LeftUpLeg,
9: LeftLeg, 10: LeftFoot, 11: LeftToeBase, 12: RightUpLeg,
13: RightLeg, 14: RightFoot, 15: RightToeBase

Upper Body (8): 0-7
Lower Body (8): 8-15

Skeleton edges (15 bones):
  (0,1) (0,2) (2,3) (3,4) (0,5) (5,6) (6,7)
  (0,8) (8,9) (9,10) (10,11) (0,12) (12,13) (13,14) (14,15)
```
