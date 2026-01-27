# Next Model Ideas (2026-01-27)

## Problem Analysis

### Key Findings

| Model | Full Body | Upper Body | Lower Body |
|------|-----------|------------|------------|
| **Baseline** | **41.37mm** | 29.42mm | **53.31mm** |
| **ViT v3** | 45.34mm | **23.49mm** ⭐ | 67.19mm |

**The Paradox of ViT v3:**
- Upper Body: **-5.93mm better** than Baseline (best!)
- Lower Body: **+13.88mm worse** than Baseline
- Most of the Full Body difference (+3.97mm) comes from Lower Body

### Root Cause Analysis

```
HMD information composition:
  ✓ Head position (3D)    → Beneficial for Upper Body
  ✓ Right hand (3D)       → Beneficial for Upper Body
  ✓ Left hand (3D)        → Beneficial for Upper Body
  ✗ No Lower body info    → Disadvantageous for Lower Body

Baseline's strengths:
  - HeatmapEncoder uniformly encodes 2D info of the entire body into Z
  - Z[64] provides balanced representation for all joints
  - HMD serves only as auxiliary (Z + HMD concat)

ViT v3's weaknesses:
  - HMD Cross-Attention provides depth reference biased toward upper body
  - Joint Tokens learn depth based on HMD reference
  - Lower Body estimates depth only via Self-Attention → unstable without HMD anchor
```

---

## Option 1: Upper-Lower Decoupled Network (Recommended) ⭐

### Core Idea
- Upper Body: ViT v3 style (leveraging HMD, achieved 23.49mm)
- Lower Body: Baseline style (without HMD, Z vector based)
- Use optimized architecture separately for each body part

### Architecture

```
                    Backbone feat [2048, 8, 8]
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  Upper Body Branch  │       │  Lower Body Branch  │
    │  (ViT v3 style)     │       │  (Baseline style)   │
    │                     │       │                     │
    │  - Spatial Tokens   │       │  - Deconv           │
    │  - Joint Queries[8] │       │  - Heatmap[8]       │
    │  - Self-Attention   │       │  - HeatmapEncoder   │
    │  - HMD Cross-Attn   │       │  - Z vector [64]    │
    │  - Heatmap Recon    │       │  - PoseDecoder      │
    └─────────────────────┘       └─────────────────────┘
              │                               │
              │  Upper pose [8, 3]            │  Lower pose [8, 3]
              │                               │
              └───────────────┬───────────────┘
                              │
                        Concat → [16, 3]
                              │
                   (Optional) Refinement Layer
                              │
                        3D Pose [16, 3]
```

### Joint Partition

**Upper Body (8 joints):**
- 0: Head
- 1: Neck
- 2: R_Shoulder
- 3: R_Elbow
- 4: L_Wrist (HMD)
- 5: L_Elbow
- 6: L_Shoulder
- 7: R_Wrist (HMD)

**Lower Body (8 joints):**
- 8: R_Knee
- 9: R_Ankle
- 10: R_Foot
- 11: L_Hip
- 12: L_Knee
- 13: L_Ankle
- 14: L_Foot
- 15: Pelvis

### Expected Results

```
Upper Body: 23.49mm (ViT v3 level)
Lower Body: 53.31mm (Baseline level)
Full Body:  (23.49 * 8 + 53.31 * 8) / 16 = 38.40mm

→ ~3mm improvement over Baseline (41.37mm)!
```

### Advantages
- Uses optimized architecture for each body part
- Applies HMD Cross-Attention only to Upper Body where HMD info is available
- Reuses the proven Baseline architecture for Lower Body

### Disadvantages
- Increased model complexity (two branches)
- Potential consistency issues at Upper/Lower boundary joints

---

## Option 2: Hierarchical Pose Estimation (v2 - Cross-Attention Based)

### Core Idea
- Stage 1: Predict Upper Body first (anchor)
- Stage 2: Predict Lower Body conditioned on Upper pose + image
- **Learn 2D positions via Cross-Attention** (eliminates Self-Attention redundancy)
- **Ensure 2D supervision via Heatmap Recon Loss**

### Architecture

```
                    Backbone feat [2048, 8, 8]
                              │
                              ▼
                    Spatial Tokens [64, D]  ← Generated once (shared)
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  Stage 1: Upper Body         │ │  Stage 2: Lower Body         │
│                              │ │                              │
│  Upper Queries [8, D]        │ │  Lower Queries [8, D]        │
│           │                  │ │           │                  │
│           ▼                  │ │           ▼                  │
│  Cross-Attn (2D):            │ │  Cross-Attn (2D):            │
│    Q: Upper Queries          │ │    Q: Lower Queries          │
│    K/V: Spatial Tokens       │ │    K/V: Spatial Tokens       │
│           │                  │ │           │                  │
│           ▼                  │ │           ▼                  │
│  Upper Tokens [8, D]─────────┼─┼─→ Cross-Attn (Upper):       │
│           │                  │ │      Q: Lower Tokens         │
│           ├─→ Heatmap Recon  │ │      K/V: Fused Upper        │
│           │   (2D loss) ✅   │ │           │                  │
│           ▼                  │ │           ├─→ Heatmap Recon  │
│  HMD Cross-Attn              │ │           │   (2D loss) ✅   │
│           │                  │ │           ▼                  │
│           ▼                  │ │  3D Head                     │
│  3D Head → Upper [8,3]───────┼─┘           │                  │
│           │                  │              ▼                  │
│           ▼                  │   Lower Pose [8, 3]           │
│  Pose Embed [8, D]───────────┼─→ (Passed to Fusion Layer)    │
│                              │                                │
└──────────────────────────────┘ └─────────────────────────────┘
                                              │
                                              ▼
                              Concat → Full Pose [16, 3]
```

### Fusion Layer (Combining Upper Info)

```python
# Upper Joint Tokens [8, D] - image features
# Upper Pose Embedding [8, D] - 3D coordinate info
fused = self.fusion_linear(
    torch.cat([upper_joint_tokens, upper_pose_embedding], dim=-1)
)  # [8, 2D] → [8, D]
```

### Loss Composition

| Loss | Target | Role |
|------|------|------|
| `loss_upper_heatmap_recon` | Upper Heatmap [8] | **Upper 2D position learning** |
| `loss_lower_heatmap_recon` | Lower Heatmap [8] | **Lower 2D position learning** |
| `loss_pose_l2norm` | Full Pose [16, 3] | 3D coordinates |
| `loss_cosine_similarity` | Full Pose | Direction |
| `loss_limb_length` | Full Pose | Bone length |
| `loss_hmd` | HMD reconstruction | HMD consistency |

### Advantages
- **Spatial Tokens sharing**: Eliminates redundant computation
- **2D learning via Cross-Attention**: Self-Attention [72x72] → Cross-Attention [8x64]
- **Heatmap Recon Loss**: Ensures 2D supervision
- **Maintains Hierarchical structure**: Lower references Upper results
- Lower Body leverages both types of information:
  1. Spatial Tokens → Image 2D positions
  2. Fused Upper → 3D relationship with Upper Body

### Disadvantages
- Upper Body errors may propagate to Lower Body
- Requires careful gradient flow design between stages (detach decision)

---

## Option 3: Symmetric Prior Enhancement

### Core Idea
- Leverage left/right joint symmetry as a prior
- In standing pose, left/right depth should be similar

### Architecture

```
Joint Tokens [16, D]
       │
       ▼
┌─────────────────────────────────────┐
│  Symmetric Attention Layer          │
│                                     │
│  Symmetric pairs:                   │
│    - L_Shoulder ↔ R_Shoulder       │
│    - L_Elbow ↔ R_Elbow             │
│    - L_Wrist ↔ R_Wrist             │
│    - L_Hip ↔ R_Hip                 │
│    - L_Knee ↔ R_Knee               │
│    - L_Ankle ↔ R_Ankle             │
│    - L_Foot ↔ R_Foot               │
│                                     │
│  Attention: Q=Left, K/V=Right      │
│           + Q=Right, K/V=Left      │
└─────────────────────────────────────┘
       │
       ▼
  Refined Tokens
       │
       ▼
  3D Pose Head
```

### Symmetric Loss

```python
def symmetric_depth_loss(pose_3d):
    pairs = [(2, 6), (3, 5), (4, 7), (8, 12), (9, 13), (10, 14)]  # L-R pairs
    loss = 0
    for l_idx, r_idx in pairs:
        # Depth difference penalty
        depth_diff = torch.abs(pose_3d[:, l_idx, 2] - pose_3d[:, r_idx, 2])
        loss += depth_diff.mean()
    return loss
```

### Advantages
- Simple to implement
- Can be added to existing models
- Improves left-right consistency

### Disadvantages
- Penalizes asymmetric poses (e.g., standing on one leg)
- Requires dynamic weight depending on action

---

## Option 4: Lower Body Pelvis Proxy

### Core Idea
- Provide pseudo-anchor for Lower Body which lacks HMD
- Estimate Pelvis position from Upper Body results
- Use Pelvis as "HMD" role for Lower Body

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Upper Body + Pelvis                               │
│                                                             │
│  Backbone feat → ViT v3 style                              │
│        │                                                    │
│        ↓                                                    │
│  Joint Queries [9] (8 upper + 1 pelvis)                    │
│        │                                                    │
│        ↓                                                    │
│  Self-Attention + HMD Cross-Attention                      │
│        │                                                    │
│        ↓                                                    │
│  Upper pose [8, 3] + Pelvis [1, 3]                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Lower Body (anchored by Pelvis)                   │
│                                                             │
│  Lower Joint Queries [7] (excluding pelvis)                │
│        │                                                    │
│        ↓                                                    │
│  Pelvis Cross-Attention:                                   │
│    Q: Lower Queries [7]                                    │
│    K/V: Pelvis token [1]                                   │
│        │                                                    │
│        ↓                                                    │
│  Lower pose [7, 3]                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Full pose [16, 3]
```

### Pelvis Estimation Method

```python
# Option A: Learnable query
pelvis_query = nn.Parameter(torch.randn(1, 1, D))

# Option B: Estimate from Upper pose
def estimate_pelvis(upper_pose):
    # Pelvis ≈ midpoint of shoulders + offset
    l_shoulder = upper_pose[:, 6]  # L_Shoulder
    r_shoulder = upper_pose[:, 2]  # R_Shoulder
    neck = upper_pose[:, 1]        # Neck

    shoulder_mid = (l_shoulder + r_shoulder) / 2
    # Pelvis is roughly below neck
    pelvis = shoulder_mid + (shoulder_mid - neck) * 1.5
    return pelvis
```

### Advantages
- Provides depth anchor for Lower Body even without HMD
- Pelvis serves as a natural body center

### Disadvantages
- Pelvis estimation errors propagate to Lower Body
- 2-stage training complexity

---

## Implementation Priority

| Rank | Option | Expected Effect | Complexity | Risk | Status |
|------|--------|----------|--------|--------|------|
| 1 | **Option 1: Decoupled** | ~38mm | Medium | Low | ✅ Implementation complete |
| 2 | **Option 2: Hierarchical** | ~39mm | Medium | Medium | ✅ Implementation complete |
| 3 | Option 3: Symmetric | +1~2mm | Low | Low | Pending |
| 4 | Option 4: Pelvis Proxy | ~40mm | Medium | Medium | Pending |
| 5 | **Option 5: ViT v6 (SPT+LSA)** | ~43mm | Low | Low | ✅ Implementation complete |

---

## Option 1 Implementation Complete (2026-01-27)

### Implementation Files

| File | Purpose |
|------|------|
| `custom_egopose_decoupled_head.py` | Head implementation |
| `HMD_xregopose_decoupled_small_config.py` | Smoke test config |
| `HMD_xregopose_decoupled_full_config.py` | Full training config |

### Smoke Test Results (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 637.00mm | 173.13mm | 1100.86mm |
| 2 | 531.01mm | 124.07mm | 937.95mm |

✅ Model trains normally
✅ Confirmed improvement in Upper/Lower body separately
✅ All loss components verified working

### Next Steps

1. Full training (10 epochs)
2. Result analysis
3. Implement Options 3-4 if needed

---

## Option 2 Implementation Complete (2026-01-27)

### Implementation Files

| File | Purpose |
|------|------|
| `custom_egopose_hierarchical_head.py` | Head implementation |
| `HMD_xregopose_hierarchical_small_config.py` | Smoke test config |
| `HMD_xregopose_hierarchical_full_config.py` | Full training config |

### Smoke Test Results (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 264.39mm | 177.70mm | 351.09mm |
| 2 | 194.68mm | 137.30mm | 252.06mm |

✅ Model trains normally
✅ Hierarchical structure (Lower references Upper) verified working
✅ Cross-Attention based 2D learning + Heatmap Recon Loss verified working

---

---

## Option 5: ViT Lifting v6 (Small Dataset Optimized) ⭐ New

### Background

**Problem**: The EgoPose dataset (210K) is very small for ViT to learn effectively
- ViT minimum requirement: 14M+ (ImageNet-21k)
- EgoPose: 210K (**1.5%** of the requirement)
- Root cause of ViT v3's validation spike and training instability

**Solution**: Apply ViT techniques designed for small datasets
- **SPT (Shifted Patch Tokenization)**: Injects locality inductive bias
- **LSA (Locality Self-Attention)**: Learnable temperature + Diagonal masking
- **Model reduction**: Prevents overfitting

### Key Techniques

#### 1. Shifted Patch Tokenization (SPT)

```python
# Existing: Simple projection
spatial_tokens = self.proj(backbone_feat)  # [B, D, 8, 8]

# SPT: Locality enhancement via 5-directional shift
x_left  = F.pad(x, (1, 0, 0, 0))[:, :, :, :W]   # Left shift
x_right = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]   # Right shift
x_up    = F.pad(x, (0, 0, 1, 0))[:, :, :H, :]   # Up shift
x_down  = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]   # Down shift

x_concat = torch.cat([x, x_left, x_right, x_up, x_down], dim=1)  # [B, 5C, H, W]
spatial_tokens = self.proj(x_concat)  # 5x expanded receptive field
```

**Effect**: Directly injects neighbor pixel info into tokens → mimics CNN's locality bias

#### 2. Locality Self-Attention (LSA)

```python
class LocalitySelfAttention:
    def __init__(self):
        # Key 1: Learnable temperature (low initial value → sharp attention)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # Attention scores
        attn = (Q @ K.T) / sqrt(D)
        attn = attn / self.temperature.clamp(min=0.1)  # Sharp attention

        # Key 2: Diagonal masking (removes self-relation)
        diag_mask = torch.eye(N).bool()
        attn = attn.masked_fill(diag_mask, float('-inf'))

        return softmax(attn) @ V
```

**Effect**:
- Learnable temperature: Sharp attention initially → focuses on neighbors
- Diagonal masking: Ignores self → uses only surrounding information

#### 3. Model Reduction

| Item | v3 (existing) | v6 (reduced) | Reduction |
|------|----------|----------|--------|
| embed_dim | 256 | 128 | 50% |
| num_heads | 8 | 4 | 50% |
| num_layers | 4 | 2 | 50% |
| mlp_ratio | 4.0 | 2.0 | 50% |
| dropout | 0.1 | 0.2 | +100% |
| **Total params** | ~5M | ~1.2M | **76%↓** |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Backbone feat [2048, 8, 8]                                 │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  SPT (Shifted Patch Tokenization)       │  ← NEW         │
│  │  5-dir shift: [2048,8,8] → [10240,8,8] │                │
│  │  → proj → [128, 8, 8]                   │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Depth-wise Conv Embedding              │  ← NEW         │
│  │  DWConv(3×3) → BN → GELU               │                │
│  │  Local feature enhancement              │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Spatial Tokens [64, 128] + Joint Queries [16, 128]         │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  LSA (Locality Self-Attention) × 2      │  ← NEW         │
│  │  - Learnable temperature (init=0.5)     │                │
│  │  - Diagonal masking (removes self-relation)│             │
│  │  - mlp_ratio=2.0 (reduced)              │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Joint Tokens [16, 128]                                     │
│         │                                                   │
│         ├── Heatmap Decoder (Reconstruction)                │
│         │   (Enforced 2D information injection)             │
│         ↓                                                   │
│  HMD Cross-Attention [16×3]                                 │
│         ↓                                                   │
│  3D Pose Head → [16, 3]                                     │
└─────────────────────────────────────────────────────────────┘
```

### Summary of Changes vs v3

| Item | v3 | v6 | Effect |
|------|-----|-----|------|
| Tokenization | Linear proj | **SPT** | Locality bias ↑ |
| Self-Attention | Standard | **LSA** | Sharp local attention |
| embed_dim | 256 | **128** | Overfitting ↓ |
| num_layers | 4 | **2** | Parameters ↓ |
| mlp_ratio | 4.0 | **2.0** | Parameters ↓ |
| dropout | 0.1 | **0.2** | Regularization ↑ |
| DWConv | ❌ | **✅** | Local features |

### Expected Effect

```
v3 problems:
- Epoch 2, 5 validation spike
- Training 256-dim, 4-layer with 210K data → overfitting
- Global attention is unstable on small dataset

v6 expectations:
- SPT provides locality bias → stabilizes early training
- LSA provides sharp local attention → learns neighboring joint relationships
- Model reduction prevents overfitting
- Reduced validation spikes → consistent performance
```

### Reference Papers

- [Vision Transformer for Small-Size Datasets (AAAI 2022)](https://arxiv.org/abs/2112.13492)
- [Depth-Wise Convolutions in ViTs (Neural Networks 2024)](https://www.sciencedirect.com/science/article/pii/S0925231024017697)

---

## Option 5 Implementation Complete (2026-01-27)

### Implementation Files

| File | Purpose |
|------|------|
| `custom_egopose_vit_lifting_head_v6.py` | Head implementation (SPT + LSA + reduced model) |
| `HMD_xregopose_vit_lifting_v6_small_config.py` | Smoke test config |
| `HMD_xregopose_vit_lifting_v6_full_config.py` | Full training config |

### Smoke Test Results (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body | Improvement |
|-------|-----------|------------|------------|--------|
| 1 | 248.77mm | 185.22mm | 312.32mm | - |
| 2 | 207.42mm | 157.40mm | 257.44mm | -16.6% |

✅ Model trains normally
✅ SPT (Shifted Patch Tokenization) verified working
✅ LSA (Locality Self-Attention) verified working
✅ Depth-wise Conv Embedding verified working
✅ Reduced model (embed_dim=128, num_layers=2) verified working

### v3 vs v6 Smoke Test Comparison (Reference)

| Item | v3 | v6 | Note |
|------|-----|-----|------|
| embed_dim | 256 | 128 | 50% reduction |
| num_layers | 4 | 2 | 50% reduction |
| Params (estimated) | ~5M | ~1.2M | 76% reduction |
| GPU Memory | ~7GB | ~6GB | ~14% savings |

### Next Steps

1. Full training (10 epochs)
2. Compare with v3 (verify reduction of validation spikes)
3. If results are good, individual SPT/LSA ablation study

---

## Reference: Joint Indices (xRegopose)

```
Upper Body (8):
  0: Head
  1: Neck
  2: R_Shoulder
  3: R_Elbow
  4: L_Wrist
  5: L_Elbow
  6: L_Shoulder
  7: R_Wrist

Lower Body (8):
  8: R_Knee
  9: R_Ankle
  10: R_Foot
  11: L_Hip
  12: L_Knee
  13: L_Ankle
  14: L_Foot
  15: Pelvis
```
