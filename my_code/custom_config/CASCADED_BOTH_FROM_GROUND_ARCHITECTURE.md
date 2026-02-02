# Cascaded Refinement + Both From Ground V2b Architecture

> Best HMD-Deployable Result: **34.24mm** (Full Body), **22.04mm** (Upper Body), **46.45mm** (Lower Body)

## V2b Optimizations

| Optimization | V1 | V2b | Effect |
|--------------|-----|-----|--------|
| HeatmapDecoder | Linear (40M) | **EfficientDecoder (1.35M)** | -96.6% params, +2.21mm |
| Training Epochs | 10 | **20** | -1.43mm improvement |
| Best Epoch | 10 | 19 | Extended convergence |
| Checkpoint Size | 365 MB | **221 MB** | -39.5% smaller |

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              Cascaded Refinement + Both From Ground V2b (34.24mm)               │
│                      Best HMD-Deployable Architecture                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                          │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────────┐  │
│  │   Image      │    │  Enhanced HMD Info (12-dim)                          │  │
│  │  256x256x3   │    │  ┌─────────────────────────────────────────────────┐ │  │
│  └──────┬───────┘    │  │ Base HMD (9-dim):                               │ │  │
│         │            │  │   right_local(3) + left_local(3) +              │ │  │
│         │            │  │   hand_dist(1) + right_dist(1) + left_dist(1)   │ │  │
│         │            │  ├─────────────────────────────────────────────────┤ │  │
│         │            │  │ Ground-Based Heights (3-dim): HMD-Deployable    │ │  │
│         │            │  │   head_from_ground(1)      <- room setup        │ │  │
│         │            │  │   left_hand_from_ground(1) <- controller        │ │  │
│         │            │  │   right_hand_from_ground(1)<- controller        │ │  │
│         │            │  └─────────────────────────────────────────────────┘ │  │
│         │            └───────────────────────────┬──────────────────────────┘  │
└─────────┼────────────────────────────────────────┼──────────────────────────────┘
          │                                        │
          ▼                                        │
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Coarse Pose Estimation                                                │
│                                                                                 │
│  ┌─────────────────┐                                                            │
│  │ ResNet-101      │                                                            │
│  │ (COCO pretrained)│                                                           │
│  └────────┬────────┘                                                            │
│           │ [B, 2048, 8, 8]                                                     │
│           ▼                                                                     │
│  ┌─────────────────┐                                                            │
│  │ Deconv Layers   │                                                            │
│  │ + Upsample      │                                                            │
│  └────────┬────────┘                                                            │
│           │ [B, 256, 47, 47]                                                    │
│           ▼                                                                     │
│  ┌─────────────────┐                                                            │
│  │ Final Conv 1x1  │                                                            │
│  └────────┬────────┘                                                            │
│           │ [B, 16, 47, 47]                                                     │
│           ▼                                                                     │
│  ┌─────────────────┐     ┌──────────────┐                                       │
│  │   Heatmaps      │────>│ EnhancedEncoder│<──── HMD Info (12-dim)              │
│  │  [B, 16, 47, 47]│     │  Conv->GAP    │       │                              │
│  └─────────────────┘     │  + HMD->Linear│       │                              │
│           │              └───────┬──────┘       │                              │
│           │                      │ Z [B, 64]    │                              │
│           │                      ▼              │                              │
│           │              ┌──────────────┐       │                              │
│           │              │  Z + HMD_emb │<──────┘                              │
│           │              │  [B, 64]     │  Linear(12->64)                       │
│           │              └───────┬──────┘                                       │
│           │                      │                                              │
│           │         ┌────────────┼────────────┐                                 │
│           │         ▼            ▼            ▼                                 │
│           │  ┌────────────┐ ┌──────────┐ ┌──────────────┐                       │
│           │  │PoseDecoder │ │EfficientHM│ │HMD Recon     │                      │
│           │  │ LinearModel│ │Decoder   │ │ (base 9-dim) │                       │
│           │  └─────┬──────┘ │(1.35M)   │ └──────────────┘                       │
│           │        │        └────┬─────┘                                        │
│           │        ▼             │                                              │
│           │  ┌──────────────┐    │     Stage 1 Losses:                          │
│           │  │ Coarse 3D    │    │     loss_kpt, loss_pose_l2norm,              │
│           │  │ [B, 16, 3]   │────┼───> loss_cosine, loss_limb_length,           │
│           │  └──────┬───────┘    │     loss_heatmap_recon, loss_hmd             │
└───────────┼─────────┼────────────┼──────────────────────────────────────────────┘
            │         │            │
            │         ▼            │
┌───────────┼─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Per-Joint Refinement with HMD Context                                 │
│           │         │                                                           │
│           │    Coarse Pose                                                      │
│           │    [B, 16, 3]                                                       │
│           │         │                                                           │
│           ▼         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Per-Joint Feature Assembly (355-dim per joint)                         │   │
│  │  [coarse(3) + spatial(64) + Z(64) + pose(128) + kin(64) + hmd(32)]     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                            │                                                │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RefinementMLP -> delta                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Refined Pose = Coarse Pose + delta                                  │   │
│  │  [B, 16, 3]                                                         │   │
│  │                                                                     │   │
│  │  Stage 2 Losses: loss_pose_l2norm_refined, loss_bone_length,        │   │
│  │                  loss_symmetry                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┼────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Refined 3D Pose [B, 16, 3]                                             │   │
│  │                                                                         │   │
│  │  V2b Results: Full Body 34.24mm | Upper Body 22.04mm | Lower Body 46.45mm│  │
│  │  vs Baseline: -7.13mm (-17.2%)                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Coarse Pose Estimation (Detailed)

Stage 1 follows the xR-EgoPose baseline architecture with Enhanced HMD integration.

### Step 1: Backbone Feature Extraction

**Purpose**: Extract visual features from input image

```
Input Image [B, 3, 256, 256]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  ResNet-101 (COCO Pretrained)                               │
│                                                             │
│  Layer 1: Conv + MaxPool -> [B, 256, 64, 64]               │
│  Layer 2: Residual Blocks -> [B, 512, 32, 32]              │
│  Layer 3: Residual Blocks -> [B, 1024, 16, 16]             │
│  Layer 4: Residual Blocks -> [B, 2048, 8, 8]               │
│                                                             │
│  Output: backbone_feat [B, 2048, 8, 8]                     │
│  Parameters: 42.5M                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### Step 2: Heatmap Generation (Deconv + Upsample)

**Purpose**: Generate per-joint 2D probability heatmaps

```
backbone_feat [B, 2048, 8, 8]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Deconvolution Layers                                       │
│                                                             │
│  deconv1: ConvTranspose2d(2048, 256, k=4, s=2) + BN + ReLU │
│           -> [B, 256, 16, 16]                              │
│  deconv2: ConvTranspose2d(256, 256, k=4, s=2) + BN + ReLU  │
│           -> [B, 256, 32, 32]                              │
│  deconv3: ConvTranspose2d(256, 256, k=4, s=2) + BN + ReLU  │
│           -> [B, 256, 64, 64]                              │
│                                                             │
│  add_deconv (upsample to 47x47):                           │
│  ConvTranspose2d(256, 256, k=3, s=1) + AdaptiveAvgPool(47) │
│           -> [B, 256, 47, 47]                              │
│                                                             │
│  Final Conv: Conv2d(256, 16, k=1) -> [B, 16, 47, 47]       │
│                                                             │
│  Output: heatmaps [B, 16, 47, 47]  (16 joints)             │
└─────────────────────────────────────────────────────────────┘
```

**Heatmap Interpretation**:
- Each of 16 channels is a 47×47 probability map for one joint
- Higher values indicate higher probability of joint presence
- Used for: (1) 2D supervision, (2) Grid sampling in Stage 2

---

### Step 3: Enhanced Encoder (Heatmap + HMD Fusion)

**Purpose**: Encode heatmaps and HMD info into compact latent Z

```
heatmaps [B, 16, 47, 47]          HMD Info [B, 12]
       │                                │
       ▼                                │
┌──────────────────────┐                │
│  Encoder Convs       │                │
│  Conv2d(16, 32, k=3) │                │
│  -> MaxPool(2)       │                │
│  Conv2d(32, 64, k=3) │                │
│  -> MaxPool(2)       │                │
│  Conv2d(64, 128, k=3)│                │
│  -> AdaptiveAvgPool  │                │
│  -> Flatten          │                │
│  -> Linear(128, 64)  │                │
└──────────┬───────────┘                │
           │ hm_feat [B, 64]            │
           │                            ▼
           │                   ┌────────────────┐
           │                   │ HMD Embedding  │
           │                   │ Linear(12, 64) │
           │                   │ ReLU           │
           │                   └───────┬────────┘
           │                           │ hmd_feat [B, 64]
           ▼                           ▼
      ┌────────────────────────────────────────┐
      │  Element-wise Addition                 │
      │  Z = hm_feat + hmd_feat               │
      │  [B, 64]                              │
      └────────────────────────────────────────┘
```

**Why Fusion Here?**
- HMD provides head/hand positions that are KNOWN
- Ground heights give absolute depth reference
- Early fusion allows Z to encode both visual AND sensor info

**Why Element-wise Addition Instead of Concatenation?**

```
Addition:       Z[64] + HMD_emb[64] = Z_fused[64]   ← Used
Concatenation:  cat(Z[64], HMD_emb[64]) = Z_fused[128]
```

| Reason | Explanation |
|--------|-------------|
| **Dimension Preservation** | Downstream modules (PoseDecoder, HeatmapDecoder) remain unchanged. Concat would require doubling their input dimensions. |
| **Shared Latent Space** | Both `z` and `hmd_emb` are projected to same 64-dim space, forcing compatible representations where each dimension has consistent meaning. |
| **Residual-like Fusion** | HMD acts as correction/bias to visual features: `Z_fused = Z_visual + HMD_bias`. Visual features dominate, HMD provides supplementary depth cues. |
| **Empirical Evidence** | Concat-based fusion (HMD Attention Fusion: 44.65mm) performed worse than addition-based (Cascaded V2b: 34.24mm). |

**Intuition**: HMD provides **complementary** depth hints, not **independent** features. Addition is sufficient for this type of fusion.

---

### Step 4: PoseDecoder (Z → 3D Pose)

**Purpose**: Decode latent Z to 3D joint coordinates

```
Z [B, 64]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PoseDecoder (LinearModel)                                  │
│                                                             │
│  Linear(64, 512) + ReLU + Dropout(0.5)                     │
│  Linear(512, 512) + ReLU + Dropout(0.5)                    │
│  Linear(512, 48)  # 16 joints × 3 coords                   │
│                                                             │
│  Reshape: [B, 48] -> [B, 16, 3]                            │
│                                                             │
│  Output: coarse_pose [B, 16, 3]                            │
│  Parameters: ~0.5M                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### Step 5: EfficientHeatmapDecoder (V2b - Z → Heatmaps)

**Purpose**: Reconstruct heatmaps from Z for self-supervision

**V1 (Linear Decoder - 40M params)**:
```
Z [B, 64] -> Linear(64, 35344) -> reshape -> [B, 16, 47, 47]
           (64 × 35344 = 2.26M just for this layer!)
```

**V2b (EfficientDecoder - 1.35M params)**:
```
Z [B, 64]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  EfficientHeatmapDecoder (Conv-based)                       │
│                                                             │
│  Linear(64, 256×3×3) -> reshape -> [B, 256, 3, 3]          │
│                                                             │
│  ConvTranspose2d(256, 128, k=4, s=2) + BN + ReLU           │
│           -> [B, 128, 8, 8]                                │
│  ConvTranspose2d(128, 64, k=4, s=2) + BN + ReLU            │
│           -> [B, 64, 18, 18]                               │
│  ConvTranspose2d(64, 32, k=4, s=2) + BN + ReLU             │
│           -> [B, 32, 38, 38]                               │
│  ConvTranspose2d(32, 16, k=4, s=2) + BN + ReLU             │
│           -> [B, 16, 78, 78]                               │
│  AdaptiveAvgPool2d(47, 47)                                 │
│           -> [B, 16, 47, 47]                               │
│                                                             │
│  Output: reconstructed_heatmaps [B, 16, 47, 47]            │
│  Parameters: 1.35M (96.6% reduction!)                       │
└─────────────────────────────────────────────────────────────┘
```

**Why EfficientDecoder Works Better**:
1. Preserves 2D spatial structure through progressive upsampling
2. Conv kernels share weights → better generalization
3. Inductive bias: local patterns are important for heatmaps

---

### Step 6: HMD Reconstruction

**Purpose**: Reconstruct base HMD (9-dim) for self-supervision

```
Z [B, 64]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  HMD Reconstructor                                          │
│                                                             │
│  Linear(64, 9)                                             │
│                                                             │
│  Output: reconstructed_hmd [B, 9]                          │
│  (only base 9-dim, not ground heights)                     │
└─────────────────────────────────────────────────────────────┘
```

**Why Only Base HMD (9-dim) is Reconstructed, Not Ground Heights (3-dim)?**

| Component | Reconstructed? | Reason |
|-----------|----------------|--------|
| Base HMD (9-dim) | ✅ Yes | Derived from pose (head-hand relationships) |
| Ground Heights (3-dim) | ❌ No | External sensor measurements |

**Base HMD (9-dim)** - Reconstructed:
- `right_local`, `left_local`, `hand_dist`, `right_dist`, `left_dist`
- These are **derived from the pose** (head-hand spatial relationships)
- Reconstructing them forces Z to encode pose-relevant information
- Acts as **self-supervision**: if Z can reconstruct HMD, it has learned body structure

**Ground Heights (3-dim)** - NOT Reconstructed:
- `head_from_ground`, `left_hand_from_ground`, `right_hand_from_ground`
- These are **external sensor measurements** (room setup, floor detection)
- They provide **absolute reference** that cannot be inferred from the image alone
- No point reconstructing: model should USE them as input, not predict them

```
┌─────────────────────────────────────────────────────────────┐
│  HMD Info (12-dim)                                          │
│                                                             │
│  Base (9-dim) ──► Encoder ──► Z ──► Reconstruct ──► Loss   │
│  "Relative measurements - can be inferred from pose"       │
│                                                             │
│  Ground (3-dim) ──► Encoder ──► Z ──► (no reconstruction)  │
│  "Absolute measurements - external reference, just use it" │
└─────────────────────────────────────────────────────────────┘
```

**Analogy**:
- Base HMD = "relative measurements" (hand position relative to head)
- Ground heights = "absolute measurements" (height from floor)

You don't ask the model to predict floor height from an image - you give it as privileged information to help resolve depth ambiguity.

---

### Stage 1 Loss Functions

| Loss | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `loss_kpt` | 1000 | MSE(pred_heatmap, gt_heatmap) | 2D joint localization |
| `loss_pose_l2norm` | 1.0 | L2(coarse_pose, gt_pose) | 3D pose accuracy |
| `loss_cosine_similarity` | 0.1 | 1 - cos(pred_dir, gt_dir) | Bone direction |
| `loss_limb_length` | 0.25 | \|pred_len - gt_len\| | Limb proportions |
| `loss_heatmap_recon` | 500 | MSE(recon_hm, gt_hm) | Encoder quality |
| `loss_hmd` | 1.0 | MSE(recon_hmd, gt_hmd) | HMD encoding |

**Total Stage 1 Loss**:
```python
loss_stage1 = (1000 * loss_kpt +
               1.0 * loss_pose_l2norm +
               0.1 * loss_cosine +
               0.25 * loss_limb +
               500 * loss_hm_recon +
               1.0 * loss_hmd)
```

---

### Stage 1 Data Flow Summary

```
Image [256×256×3]
    │
    ├─── ResNet-101 ─────────────────────────┐
    │         │                              │
    │    [2048, 8, 8]                        │
    │         │                              │
    │    Deconv+Upsample                     │
    │         │                              │
    │    [16, 47, 47] Heatmaps               │
    │         │                              │
    │    Encoder ◄──── HMD [12] ────────┐    │
    │         │                         │    │
    │      Z [64] ─────────────────────┤    │
    │         │                         │    │
    │    ┌────┼────┬────────────────────┘    │
    │    │    │    │                         │
    │    ▼    ▼    ▼                         │
    │  Pose  HM   HMD                        │
    │  Dec  Recon Recon                      │
    │    │                                   │
    │    ▼                                   │
    │  Coarse 3D [16, 3] ──────────────────► Stage 2
    │                                        │
    └────────────────────────────────────────┘
              backbone_feat [2048, 8, 8] ───► Stage 2
```

---

## Stage 2: Per-Joint Refinement (Detailed)

### Inputs from Stage 1

| Input | Shape | Description |
|-------|-------|-------------|
| Coarse Pose | [B, 16, 3] | 3D coordinates from PoseDecoder |
| Heatmaps | [B, 16, 47, 47] | Predicted joint heatmaps |
| Backbone Features | [B, 2048, 8, 8] | ResNet-101 output |
| Z Latent | [B, 64] | Encoded representation |
| HMD Info | [B, 12] | Enhanced HMD (9 base + 3 ground heights) |

---

### Step 1: Soft-Argmax 2D Coordinate Extraction

**Purpose**: Get 2D joint locations from heatmaps for grid sampling

```
Heatmaps [B, 16, 47, 47]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  soft_argmax_2d(heatmaps, temperature=1.0)                  │
│                                                             │
│  1. Flatten: [B, 16, 47, 47] -> [B, 16, 2209]              │
│  2. Softmax: probs = softmax(hm_flat / temp)               │
│  3. Expected coordinates:                                   │
│     exp_x = sum(probs * x_coords)                          │
│     exp_y = sum(probs * y_coords)                          │
│  4. Normalize to [0, 1]: coords / (size - 1)               │
│                                                             │
│  Output: coords_2d [B, 16, 2] (normalized x, y per joint)  │
└─────────────────────────────────────────────────────────────┘
```

---

### Step 2: Per-Joint Spatial Feature Extraction (Grid Sampling)

**Purpose**: Extract backbone features AT each joint's 2D location

```
Backbone Features [B, 2048, 8, 8]    coords_2d [B, 16, 2]
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Grid Sampling                                              │
│                                                             │
│  1. Convert coords to grid format:                          │
│     grid = coords_2d * 2 - 1  (normalize to [-1, 1])       │
│     grid = grid.unsqueeze(1)  -> [B, 1, 16, 2]             │
│                                                             │
│  2. Bilinear interpolation:                                 │
│     sampled = F.grid_sample(                                │
│         backbone_feat,  # [B, 2048, 8, 8]                  │
│         grid,           # [B, 1, 16, 2]                    │
│         mode='bilinear',                                    │
│         align_corners=True                                  │
│     )                                                       │
│     -> [B, 2048, 1, 16]                                    │
│                                                             │
│  3. Reshape: [B, 2048, 1, 16] -> [B, 16, 2048]             │
└─────────────────────────────────────────────────────────────┘
```

**Visual Explanation**:
```
  Backbone Feature Map (8x8)     2D Joint Coords
  ┌─────────────────────┐        (from heatmaps)
  │ # # # # # # # #     │
  │ # # o─────────────────────── Joint 0 (head)
  │ # # # # o─────────────────── Joint 4 (L.hand)
  │ # # # # # o───────────────── Joint 7 (R.hand)
  │ # # # # # # # #     │
  │ # o───────────────────────── Joint 10 (L.foot)
  │ # # # # # o───────────────── Joint 14 (R.foot)
  │ # # # # # # # #     │        ...
  └─────────────────────┘

  Each 'o' extracts 2048-dim feature via bilinear interpolation
```

**Spatial Projection**:
```python
spatial_proj = nn.Sequential(
    nn.Linear(2048, 64),
    nn.ReLU()
)
# [B, 16, 2048] -> [B, 16, 64]
```

**Actual Code Implementation**:
```python
# 1. Get 2D joint coordinates from heatmaps
coords_2d, _ = soft_argmax_2d(heatmap.detach())
# Input:  heatmap [B, 16, 47, 47]
# Output: coords_2d [B, 16, 2] - normalized (x, y) in [0, 1]

# 2. Convert to grid_sample coordinate system [-1, 1]
grid = coords_2d * 2 - 1
# [0, 1] → [-1, 1]
# Example: 0.0 → -1.0 (left/top edge)
#          0.5 →  0.0 (center)
#          1.0 →  1.0 (right/bottom edge)

# 3. Reshape for grid_sample API
grid = grid.unsqueeze(1)
# [B, 16, 2] → [B, 1, 16, 2]
# grid_sample expects: [B, H_out, W_out, 2]
# Here: H_out=1, W_out=16 (16 sampling points)

# 4. Bilinear interpolation at joint locations
sampled = F.grid_sample(
    backbone_feat,  # [B, 2048, 8, 8]
    grid,           # [B, 1, 16, 2]
    mode='bilinear',
    align_corners=True,
    padding_mode='border'
)
# Output: [B, 2048, 1, 16]

# 5. Reshape to per-joint format
sampled = sampled.squeeze(2).permute(0, 2, 1)
# [B, 2048, 1, 16] → [B, 2048, 16] → [B, 16, 2048]

# 6. Project to lower dimension
spatial_feat = self.spatial_proj(sampled)
# [B, 16, 2048] → [B, 16, 64]
```

**Why Grid Sampling Instead of GAP?**

| Approach | Description | Problem |
|----------|-------------|---------|
| **GAP (Stage 1)** | Global Average Pooling | Loses spatial detail |
| **Grid Sampling (Stage 2)** | Sample AT each joint | Preserves local appearance |

Stage 1 uses GAP which averages all spatial locations. Stage 2 compensates by extracting features **specifically where each joint is located**, providing fine-grained local context for refinement.

**Why `heatmap.detach()`?**

```python
coords_2d, _ = soft_argmax_2d(heatmap.detach())
```

- **Stops gradient** from flowing back through coords_2d to heatmap
- Treats 2D coords as "fixed" sampling locations
- Prevents unstable training from circular dependency:
  - ❌ heatmap → coords → sample → loss → heatmap (circular, unstable)
  - ✅ heatmap → coords (detached) → sample → loss (stable)

---

### Step 3: Pose Context Encoding

**Purpose**: Global pose context shared across all joints

```python
# 1. Flatten: [B, 16, 3] -> [B, 48]
# 2. Encode:
pose_encoder_net = nn.Sequential(
    nn.Linear(48, 128),
    nn.ReLU()
)
# [B, 48] -> [B, 128]
```

---

### Step 4: Kinematic Chain Feature Extraction

**Purpose**: Encode skeletal structure (bone directions + lengths)

**EGOPOSE_SKELETON (15 bones)**:
```
(0->1)  Spine2 -> Head
(0->2)  Spine2 -> LeftArm       (0->5)  Spine2 -> RightArm
(2->3)  LeftArm -> LeftForeArm  (5->6)  RightArm -> RightForeArm
(3->4)  LeftForeArm -> LeftHand (6->7)  RightForeArm -> RightHand
(0->8)  Spine2 -> LeftUpLeg     (0->12) Spine2 -> RightUpLeg
(8->9)  LeftUpLeg -> LeftLeg    (12->13) RightUpLeg -> RightLeg
(9->10) LeftLeg -> LeftFoot     (13->14) RightLeg -> RightFoot
(10->11) LeftFoot -> LeftToe    (14->15) RightFoot -> RightToe
```

**Feature Computation**:
```python
def compute_kinematic_features(pose_3d):
    kin_feats = []
    for parent, child in EGOPOSE_SKELETON:
        bone_vec = pose_3d[:, child] - pose_3d[:, parent]  # [3-dim direction]
        bone_len = torch.norm(bone_vec, dim=-1, keepdim=True)  # [1-dim length]
        kin_feats.append(torch.cat([bone_vec, bone_len], dim=-1))  # [4-dim per bone]
    return torch.cat(kin_feats, dim=-1)  # [B, 60]

# Encode:
kin_encoder = nn.Sequential(
    nn.Linear(60, 64),
    nn.ReLU()
)
# [B, 60] -> [B, 64]
```

---

### Step 5: HMD Context Encoding (Stage 2)

**Purpose**: Provide ground-based height context for depth estimation

**HMD Info Structure (12-dim)**:
```
Base HMD (9-dim):
  right_local[3], left_local[3], hand_dist[1],
  right_dist[1], left_dist[1]

Ground Heights (3-dim): <- KEY for lower body!
  head_from_ground[1]       <- "how tall is the person"
  left_hand_from_ground[1]  <- "where is left hand height"
  right_hand_from_ground[1] <- "where is right hand height"
```

**Encoder**:
```python
hmd_encoder_stage2 = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU()
)
# [B, 12] -> [B, 32]
```

---

### Step 6: Per-Joint Feature Assembly

**Purpose**: Concatenate all features for each of 16 joints

```python
# Feature Expansion (broadcast to 16 joints)
z_exp    = Z.unsqueeze(1).expand(B, 16, 64)       # [B, 16, 64]
pose_exp = pose_feat.unsqueeze(1).expand(B, 16, 128)  # [B, 16, 128]
kin_exp  = kin_feat.unsqueeze(1).expand(B, 16, 64)    # [B, 16, 64]
hmd_exp  = hmd_feat.unsqueeze(1).expand(B, 16, 32)    # [B, 16, 32]

# Concatenation per joint
joint_input = torch.cat([
    coarse_pose,    # [B, 16, 3]   <- Joint's XYZ
    spatial_feat,   # [B, 16, 64]  <- Local visual feature
    z_exp,          # [B, 16, 64]  <- Global latent
    pose_exp,       # [B, 16, 128] <- Global pose context
    kin_exp,        # [B, 16, 64]  <- Skeletal structure
    hmd_exp,        # [B, 16, 32]  <- Height context
], dim=-1)
# -> [B, 16, 355]
```

**Per-Joint Feature Vector (355-dim)**:
```
┌─────┬────────┬───────┬────────┬───────┬───────┐
│ XYZ │Spatial │   Z   │ Pose   │ Kin   │ HMD   │
│  3  │   64   │  64   │  128   │  64   │  32   │
└─────┴────────┴───────┴────────┴───────┴───────┘
Local  Local   Global  Global  Global  Global
coord  visual  latent  pose    struct  height
```

---

### Step 7: Refinement MLP (Shared Across Joints)

**Purpose**: Learn residual correction delta for each joint

```python
class RefinementMLP(nn.Module):
    def __init__(self, input_size=355, hidden_size=256, num_stage=1, p_dropout=0.5):
        super().__init__()
        self.w1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.stages = nn.ModuleList([
            RefinementBlock(hidden_size, p_dropout)
            for _ in range(num_stage)
        ])

        self.w_out = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x: [B*16, 355]
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.stages:
            y = stage(y)  # Residual block
        return self.w_out(y)  # [B*16, 3]
```

**RefinementBlock (Residual)**:
```
Input ────────────────────────────────┐
  │                                   │
  ▼                                   │
Linear(256, 256) -> BN -> ReLU -> Drop│
  │                                   │
  ▼                                   │
Linear(256, 256) -> BN -> ReLU -> Drop│
  │                                   │
  ▼                                   ▼
Output = Input + Residual <───────────┘
```

---

### Step 8: Residual Addition

**Purpose**: Apply learned correction to coarse pose

```python
refined_pose = coarse_pose + delta
# [B, 16, 3]  =  [B, 16, 3]  +  [B, 16, 3]
```

**Why Residual Design?**
- MLP only needs to learn CORRECTIONS, not full pose
- Easier optimization (delta starts near zero)
- Can only help, never hurt (worst case: delta ~= 0)
- Preserves good predictions from Stage 1

---

## Stage 2 Losses

| Loss | Weight | Description |
|------|--------|-------------|
| `loss_pose_l2norm_refined` | 1.0 | L2 distance between refined pose and GT |
| `loss_bone_length` | 0.5 | \|predicted_bone_len - gt_bone_len\| for all 15 bones |
| `loss_symmetry` | 0.1 | \|left_bone_len - right_bone_len\| for paired limbs |

---

## Why Stage 2 Improves Performance

### 1. Local Visual Features (Grid Sampling)
- Stage 1 uses Global Average Pooling -> loses spatial detail
- Stage 2 samples AT each joint -> preserves local appearance

### 2. Explicit Skeletal Constraints (Kinematic Features)
- Bone directions + lengths encode body structure
- Helps maintain anatomical plausibility

### 3. Ground Height Context (HMD Encoder Stage 2)
- `head_from_ground` provides absolute depth reference
- Critical for lower body where depth ambiguity is highest

### 4. Per-Joint Refinement
- Each joint gets specialized correction
- Shared MLP learns joint-agnostic patterns

### 5. Residual Design
- Only learns corrections -> easier optimization
- Preserves good Stage 1 predictions

---

## Results

### V2b (Best HMD-Deployable)

| Metric | V1 | V2b | Improvement |
|--------|-----|-----|-------------|
| Full Body MPJPE | 37.88mm | **34.24mm** | **-3.64mm (-9.6%)** |
| Upper Body MPJPE | 25.10mm | **22.04mm** | **-3.06mm (-12.2%)** |
| Lower Body MPJPE | 50.66mm | **46.45mm** | **-4.21mm (-8.3%)** |
| Best Epoch | 10 | 19 | Extended training |

**Comparison with Baseline (41.37mm)**:
- V2b: **-7.13mm (-17.2%)** 🏆
- V1: -3.49mm (-8.4%)

### V2 Ablation Summary

| Variant | Optimization | MPJPE | vs V1 |
|---------|--------------|-------|-------|
| V2 | EfficientDecoder | 35.67mm | -2.21mm |
| V2a | +Stronger MLP | 39.02mm | +1.14mm ❌ |
| **V2b** | **+20 Epochs** | **34.24mm** | **-3.64mm 🏆** |
| V2c | +Loss Tuning | 37.81mm | -0.07mm |

---

## Model Specifications

| Spec | V1 | V2b |
|------|-----|-----|
| Total Parameters | ~87.5M | ~48.8M |
| HeatmapDecoder | 40M (Linear) | 1.35M (Conv) |
| Checkpoint Size | 365 MB | 221 MB |
| Training Epochs | 10 | 20 |
| Training Throughput | 141.5 samples/s | 144.5 samples/s |
| Estimated Inference FPS | ~290 | ~294 |

---

## Key Files

| File | Purpose |
|------|---------|
| `custom_egopose_cascaded_refinement_head_enhanced.py` | Head implementation |
| `HMD_xregopose_cascaded_both_from_ground_v2b_full_config.py` | V2b Training config |
| `HMD_xregopose_cascaded_both_from_ground_full_config.py` | V1 Training config |
| `enhance_hmd_info.py` | EnhanceHMDInfo transform (both_from_ground mode) |

---

## EfficientHeatmapDecoder (V2b Key Component)

**Original Linear Decoder (V1)**:
```
Z [B, 64] -> Linear(64, 47*47*16) -> [B, 35344] -> reshape -> [B, 16, 47, 47]
Parameters: 64 * 35344 = 2.26M (just for projection)
Total decoder: ~40M params
```

**EfficientDecoder (V2b)**:
```
Z [B, 64] -> Linear(64, 256*3*3) -> [B, 256, 3, 3]
         -> ConvTranspose2d(256, 128, 4, 2) -> [B, 128, 8, 8]
         -> ConvTranspose2d(128, 64, 4, 2)  -> [B, 64, 18, 18]
         -> ConvTranspose2d(64, 32, 4, 2)   -> [B, 32, 38, 38]
         -> ConvTranspose2d(32, 16, 4, 2)   -> [B, 16, 78, 78]
         -> AdaptiveAvgPool2d(47)           -> [B, 16, 47, 47]
Total decoder: ~1.35M params (96.6% reduction)
```

**Why It Works Better**:
1. **Preserves spatial structure**: Conv layers maintain 2D spatial relationships
2. **Progressive upsampling**: Gradual resolution increase vs single large projection
3. **Inductive bias**: Conv kernels share weights across spatial locations
4. **Regularization**: Fewer params = less overfitting
