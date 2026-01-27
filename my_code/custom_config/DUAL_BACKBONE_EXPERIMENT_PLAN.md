# Dual Backbone Improvement Experiment Plan

## Goal

**Achieve better performance than Single COCO (41.37mm) through Dual Backbone mutual learning.**

Current problem: Dual COCO+MPII (43.26mm) is 1.89mm worse than Single COCO (41.37mm)

---

## Baseline Results (Updated: 2026-01-21)

> **Detailed results**: See `EXPERIMENT_RESULTS.md`

### Key Comparison

| Model | Full Body MPJPE | Difference | Notes |
|-------|-----------------|------------|-------|
| **Single COCO** | **41.37mm** | - | 🏆 Current best (epoch 8) |
| Dual COCO+MPII | 43.26mm | +1.89mm | Dual is worse (epoch 8) |
| Dual Warmup v2 | 45.93mm | +4.56mm | Warmup had no effect |
| Single Lifting | 45.92mm | +4.55mm | Lack of depth |

---

## Current Model Architecture

### Single COCO (41.37mm)

**Config**: `HMD_xregopose_single_coco_full_config.py`

```
┌─────────────────────────────────────────────────────────────┐
│  TopdownPoseEstimator                                       │
├─────────────────────────────────────────────────────────────┤
│  Backbone: ResNet-101 (COCO pretrained)                     │
│       ↓                                                     │
│  Head: CustomxRegoposeBaselinel1                            │
│       ├── Deconv → Heatmap [16, 47, 47]                     │
│       ├── Encoder → Z [64]                                  │
│       ├── + HMD info [9→64]                                 │
│       ├── Pose Decoder → 3D Pose [16, 3]                    │
│       └── Heatmap Decoder → Recon Heatmap                   │
└─────────────────────────────────────────────────────────────┘
```

**Loss configuration**:
| Loss | Weight | Description |
|------|--------|-------------|
| loss_kpt | 1000 | Main heatmap MSE |
| loss_heatmap_recon | 250 | Heatmap reconstruction |
| loss_pose_l2norm | 1.0 | 3D pose L2 |
| loss_cosine_similarity | 0.1 | Cosine sim |
| loss_limb_length | 0.25 | Limb length |
| loss_hmd | 1.0 | HMD reconstruction |

---

### Dual COCO+MPII (43.26mm)

**Config**: `HMD_xregopose_h5cache_coco_mpii_config.py`

```
┌─────────────────────────────────────────────────────────────┐
│  Custom_TopdownPoseEstimator                                │
├─────────────────────────────────────────────────────────────┤
│  Backbone1: ResNet-101 (COCO pretrained)                    │
│  Backbone2: ResNet-101 (MPII pretrained)                    │
│       ↓                    ↓                                │
│  feat1 [2048,8,8]     feat2 [2048,8,8]                      │
│       ↓                    ↓                                │
│  Deconv1 → Heatmap1   Deconv2 → Heatmap2                    │
│       ↓                    ↓                                │
│       └────── MSE Loss ────┘  ← loss_backbone_latant        │
│       ↓                    ↓                                │
│  Main path            Sub path (GT supervision)             │
│       ↓                                                     │
│  Encoder → Z → Pose Decoder → 3D Pose                       │
└─────────────────────────────────────────────────────────────┘
```

**Loss configuration** (Single + additions):
| Loss | Weight | Description |
|------|--------|-------------|
| loss_backbone_latant | 1.0 | **MSE(feat1, feat2)** - backbone feature alignment |
| loss_backbone_heatmap | 1.0 | Sub backbone heatmap GT supervision |
| (+ all Single losses) | | |

---

## Problem Analysis

**Observation**: Dual (43.26mm) > Single (41.37mm) - Dual is 1.89mm worse

### Problem 1: Mutual Learning Approach

1. `loss_backbone_latant = MSE(feat1, feat2)`: **Unconditionally forces** the two backbone features to be identical
2. Different feature distributions of COCO and MPII collide
3. Unique strengths of each pretrained model cancel out
4. Mutual learning enforced from epoch 1 → pretrained knowledge damaged

### Problem 2: Limitations of 3D Information Encoding in Heatmaps (Structural)

```
Backbone feat [2048, 8, 8]  ← Rich depth/texture/context information
       ↓ (Deconv)
Heatmap [16, 47, 47]        ← Only 2D positions remain (depth lost!)
       ↓ (Encoder)
Z [64]                      ← Extreme compression
       ↓
3D Pose                     ← Depth ambiguity occurs
```

**Basis**:
- Heatmaps are fundamentally **probability distributions of 2D positions** (expressing only x, y)
- 3D depth information exists only implicitly → ambiguity during lifting
- Backbone feature's depth cues are lost at the heatmap stage

**Solution direction**: Preserve depth information by keeping backbone features through a separate path

---

## Experiment Phases

### Phase 1: Progressive Warmup (Currently implemented)

**Purpose**: Introduce mutual learning while preserving pretrained knowledge

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

| Parameter | Value | Description |
|-----------|-------|-------------|
| mutual_warmup_epochs | 5 | No mutual learning |
| mutual_rampup_epochs | 10 | Linear increase 0→1 |
| max_epochs | 20 | 5+10+5 |

**Expected effect**:
- Preserves COCO/MPII individual strengths during early training
- Prevents sudden gradient collision

**Success criteria**: MPJPE improvement compared to Dual COCO+MPII (original)

---

### Phase 2: Ensemble Teacher

**Purpose**: Create a pseudo-teacher by ensembling outputs from both backbones

**Implementation location**: `custom_egopose_baselinel1_head_multi_backbone_v3.py`

**Changes**:
```python
# Current (v2)
loss_backbone_latant = MSE(feat1, feat2) * warmup_weight

# Improved (v3)
conf1 = heatmap1.max().mean()
conf2 = heatmap2.max().mean()
w1 = conf1 / (conf1 + conf2)
w2 = conf2 / (conf1 + conf2)

feat_ensemble = w1 * feat1.detach() + w2 * feat2.detach()
loss_ensemble = MSE(feat1, feat_ensemble) + MSE(feat2, feat_ensemble)
loss_backbone_latant = loss_ensemble * warmup_weight
```

**Expected effect**:
- Higher weight for the backbone with higher confidence
- Combining strengths of both pretrained models

**Experiment combinations**:
| Experiment | Warmup | Ensemble | Notes |
|------------|--------|----------|-------|
| 2-A | ✓ | ✓ | Phase 1 + Ensemble |
| 2-B | ✗ | ✓ | Ensemble only |

---

### Phase 3: Heatmap KL Divergence

**Purpose**: Apply the Deep Mutual Learning paper approach

**Implementation location**: `custom_egopose_baselinel1_head_multi_backbone_v4.py`

**Changes**:
```python
def heatmap_kl_divergence_loss(heatmap1, heatmap2, temperature=4.0):
    h1_flat = heatmap1.view(B, K, -1) / temperature
    h2_flat = heatmap2.view(B, K, -1) / temperature

    p1 = F.softmax(h1_flat, dim=-1)
    p2 = F.softmax(h2_flat, dim=-1)

    kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2 * (temperature ** 2)
```

**Hyperparameters**:
| Parameter | Candidate values | Description |
|-----------|-----------------|-------------|
| temperature | 2, 4, 8 | Degree of soft targets |
| loss_weight | 0.1, 0.5, 1.0 | KL loss weight |

**Experiment combinations**:
| Experiment | Warmup | Ensemble | KL Div | Notes |
|------------|--------|----------|--------|-------|
| 3-A | ✓ | ✗ | ✓ | Warmup + KL |
| 3-B | ✓ | ✓ | ✓ | Full combination |

---

### Phase 4: One-way KD (Optional)

**Purpose**: For when only the main backbone is used during inference

**Changes**:
```python
# KD only in Sub → Main direction
loss_kd = MSE(feat_main, feat_sub.detach())
# Sub learns only from GT (gradient blocked)
```

**Use scenarios**:
- Remove sub backbone during inference for 2x speed
- Maximize main backbone performance

---

### Phase 5: Structural Improvements (Long-term)

#### 5-A: HeatmapDecoder Optimization

**Problem**: `linear3` has 37.77M parameters (65% of Head's 61.4M)

**Solution**: Replace with Conv-based decoder
- 40M → ~1.5M (96% reduction)
- Memory: 22GB → 16GB

#### 5-B: 2D→3D Lifting (Implementation complete ✅)

**Change**:
```
Current: Heatmap → Encoder → Z[64] → Decoder → 3D
Improved: Heatmap → soft_argmax → 2D[16,2] + conf[16] → Lifting → 3D
```

**Advantages**:
- Complete removal of HeatmapDecoder (40M → 4M)
- Martinez baseline (proven approach)
- Interpretable intermediate representation
- Simultaneous Heatmap MSE + Coord MSE training

**Implementation files**:
- Head: `custom_egopose_lifting_head.py`
- Config: `HMD_xregopose_single_lifting_config.py`

**Comparison target (important!)**:

| Item | Single COCO (current best) | Single Lifting (new) |
|------|--------------------------|----------------------|
| Config | `HMD_xregopose_single_coco_full_config.py` | `HMD_xregopose_single_lifting_config.py` |
| Head | `CustomxRegoposeBaselinel1` | `CustomEgoposeLiftingHead` |
| Backbone | ResNet-101 COCO | ResNet-101 COCO (same) |
| 2D→3D | Encoder→Z[64]→Decoder | soft_argmax→Lifting |
| Head Params | ~61M | ~13M |
| MPJPE | **41.37mm** | ? (experiment needed) |

> **Note**: Lifting should be compared with **Single backbone**, not Dual!
> Comparing with Dual (43.26mm) is meaningless due to too large structural differences.

---

### Phase 6: Backbone Feature Fusion (Key Structural Improvement)

**Problem recognition**: Heatmaps are optimized for 2D position information, making 3D depth information encoding difficult

**Academic basis**:
- [Depth Ambiguity Survey](https://www.mdpi.com/2076-3417/12/20/10591): "A single 2D pose can map to multiple 3D poses"
- [EgoTAP](https://arxiv.org/html/2402.18330): "CNN encoder fails to properly preserve heatmap information"
- [Lifting by Image](https://arxiv.org/abs/2312.15636): "Image's semantic/texture information contributes to lifting"

#### 6-A: Backbone + Heatmap Latent Concat (Recommended - try first)

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├──→ GAP → FC → Z_backbone [256]  ← depth/context cues
       │                      │
       ↓ (Deconv)             │
Heatmap [16, 47, 47]          │
       ↓ (Encoder)            │
Z_heatmap [64] ← 2D position  │
       │                      │
       └──── Concat ──────────┘
              ↓
         [64 + 256 + 64(HMD)] = [384]
              ↓
         Pose Decoder → 3D Pose
```

**Role separation**:
| Component | Role |
|-----------|------|
| Z_heatmap | Precise 2D joint positions (x, y) |
| Z_backbone | Depth/texture/context (depth cues) |
| Z_hmd | Head/hand 3D positions (absolute reference) |

**Implementation location**: `custom_egopose_baselinel1_head_multi_backbone_v5.py` (or integrated into v3)

**Config**: `HMD_xregopose_h5cache_coco_mpii_backbone_fusion_config.py`

#### 6-B: 2D Coords + Backbone Feature

**Architecture**:
```
Heatmap → soft-argmax → 2D coords [16, 2] + conf [16]
                              │
Backbone feat → GAP → FC → Context [256]
                              │
              Concat ─────────┘
                 ↓
           [32 + 16 + 256 + 9(HMD)] = [313]
                 ↓
           Lifting Network → 3D Pose
```

**Advantages**: 2D coordinates are explicit, backbone directly contributes to depth resolution

---

### Experiment Combination Summary

| Phase | Improvement Direction | Implementation Status | Comparison Target |
|-------|-----------------------|-----------------------|-------------------|
| 1 | Progressive Warmup | ✅ Complete | vs Dual COCO+MPII (43.26mm) |
| 2 | Ensemble Teacher | ❌ Not implemented | vs Dual |
| 3 | KL Divergence | ❌ Not implemented | vs Dual |
| 4 | One-way KD | ❌ Not implemented | vs Dual |
| **5-A** | HeatmapDecoder Optimization | ❌ Not implemented | - |
| **5-B** | **2D→3D Lifting** | **✅ Complete (190mm - failed)** | **vs Single COCO (41.37mm)** |
| **5-C** | **Lifting + Backbone Fusion** | **✅ Complete** | **vs 5-B** |
| **5-D** | **Attention Lifting** | **❌ Not implemented** | **vs 5-C** |
| 6 | Backbone Feature Fusion (existing architecture) | ❌ Not implemented | vs Single/Dual |

---

### Phase 5-C: Lifting + Backbone Feature Fusion (✅ Implementation complete)

**Purpose**: Add backbone features to the Lifting Head to utilize depth cues

**Architecture**:
```
Backbone feat [2048, 8, 8]
       │
       ├───────────────────────────┐
       ↓ (Deconv)                  ↓ (GAP → FC)
Heatmap [16, 47, 47]          Z_backbone [256]
       ↓ (soft_argmax)             │
2D [32] + conf [16]                │ (depth cues!)
       │  ← gradient blocked       │  ← gradient flows
       └────── Concat ─────────────┘
                  ↓
       [32 + 16 + 256 + 9] = 313
                  ↓
         Lifting Network → 3D
```

**Core design**:
- `coords_2d.detach()`: Heatmap learns only 2D positions (3D loss blocked)
- `z_backbone`: Backbone learns depth cues (3D loss flows)
- Role Separation

**Implementation files**:
- Head: `custom_egopose_lifting_backbone_fusion_head.py` ✅
- Config: `HMD_xregopose_lifting_backbone_fusion_config.py` ✅

**Comparison target**: Phase 5-B Lifting (190mm) → improvement expected

---

### Phase 5-D: Attention-based Lifting (Planned)

**Purpose**: Using Cross-Attention so each joint selectively queries the necessary depth information from the backbone

**Architecture**:
```
2D coords [16, 2] → Joint queries [16, D]
                          ↓
Backbone [2048,8,8] → Spatial tokens [64, D]
                          ↓
              Cross-Attention (Q: joints, K/V: backbone)
                          ↓
              Depth-aware joints [16, D]
                          ↓
                     + HMD → 3D Pose
```

**Core design**:
```python
# Gradient flow design
coords_2d = soft_argmax(heatmaps).detach()  # 2D: gradient blocked
joint_q = embed(coords_2d)                   # Query

backbone_kv = backbone_proj(backbone_feat)   # K/V: gradient flows!

depth_joints = cross_attention(joint_q, backbone_kv, backbone_kv)
pose_3d = output(depth_joints)
```

**Advantages**:
1. Each joint **queries depth from the needed spatial location**
2. **Interpretable via `attn_weights`** showing which positions were referenced
3. **Role separation**: 2D coords=position (Query), Backbone=depth (Key/Value)

**Implementation files** (planned):
- Head: `custom_egopose_attention_lifting_head.py`
- Config: `HMD_xregopose_attention_lifting_config.py`

**Comparison target**: Phase 5-C Backbone Fusion results

**Recommended experiment order**:
1. Phase 1 (Warmup) - already implemented, experiment in progress
2. **Phase 6-A (Backbone Fusion)** - key structural improvement, can run in parallel
3. Phase 1 + 6-A combination
4. Selectively apply Phase 2, 3 based on results


## Reference Papers

1. **Deep Mutual Learning** (Zhang et al., 2017) - arXiv:1706.00384
2. **Knowledge Distillation** (Hinton et al., 2015) - arXiv:1503.02531
3. **A Simple Baseline for 3D Pose** (Martinez et al., 2017) - ICCV 2017

---

## Current Progress

- [x] Phase 1: Progressive Warmup implementation (for Dual backbone)
  - [x] Head v2 created: `custom_egopose_baselinel1_head_multi_backbone_v2.py`
  - [x] MutualLearningWarmupHook created
  - [x] Config created: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`
- [x] **Phase 5-B: 2D→3D Lifting implementation (for Single backbone)**
  - [x] **Head created: `custom_egopose_lifting_head.py`**
  - [x] **Config created: `HMD_xregopose_single_lifting_config.py`**
  - [x] **Experiment result: 190mm (failed - insufficient depth information)**
- [ ] Phase 1 experiment execution (comparison: Dual 43.26mm)
- [x] **Phase 5-C: Lifting + Backbone Fusion implementation**
  - [x] Head created: `custom_egopose_lifting_backbone_fusion_head.py`
  - [x] Config created: `HMD_xregopose_lifting_backbone_fusion_config.py`
  - [x] Smoke test completed (2026-01-21): training pipeline confirmed working
  - [ ] Full dataset training (target: < 41.37mm)
- [ ] **Phase 5-D: Attention Lifting implementation**
  - [ ] Head creation: `custom_egopose_attention_lifting_head.py`
  - [ ] Config creation: `HMD_xregopose_attention_lifting_config.py`
- [ ] Phase 2, 3 implementation (if needed)
