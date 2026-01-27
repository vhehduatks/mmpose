# ViT-Style Attention Lifting for 3D Pose Estimation (v2)

## 0. Key Insight from EfficientHeatmapDecoder

```
Core insight of EfficientHeatmapDecoder:
  The purpose of heatmap reconstruction is NOT to decode heatmaps!
  -> The purpose is to inject joint information into the latent vector Z

  Reconstruction loss regularizes Z -> Forces Z to contain joint position information
```

**Applying this principle to ViT-Style:**
- Joint tokens = Latent representation
- Heatmap Reconstruction = Injecting 2D joint information into joint tokens
- HMD Cross-Attention = Adding 3D depth information

---

## 1. Background

### Problems with Existing Attention Lifting

```
Current structure:
  Heatmap -> soft_argmax -> 2D coords [B, 16, 2]
      |
  joint_embed(coords) -> Q [B, 16, D]
      |
  Cross-Attention(Q, K=backbone, V=backbone)
      |
  Depth prediction
```

**Problems:**
1. `soft_argmax` extracts only (x, y) coordinates -> spatial feature information loss
2. Query is coordinate-based, limiting expressiveness
3. Unidirectional information flow (backbone -> joint)

---

## 2. ViTPose Reference (arxiv 2204.12484)

### ViTPose Core Structure

```
Image -> Patch Embedding -> [B, 256, D] (16x16 patches)
    |
Transformer Encoder (Self-Attention)
    |
Simple Decoder -> Heatmaps
```

### Advantages of Patch Tokens

1. **Position information**: Spatial position preserved through positional encoding
2. **Local + Global**: Full context obtained through self-attention
3. **Semantic information**: Learned feature representation

---

## 3. Proposal: ViT-Style Lifting v2 (Reconstruction Regularized)

### Core Idea

```
Instead of extracting coordinates via soft_argmax -> Use Learnable Joint Queries
Instead of Cross-Attention -> Self-Attention (bidirectional joint <-> patch)
Heatmap Reconstruction -> Injecting 2D joint information into joint tokens
HMD Cross-Attention -> Adding 3D depth reference
```

### Structure (v2)

```
+-------------------------------------------------------------+
|                                                             |
|  Backbone (ResNet) -> Feature [B, 2048, 8, 8]               |
|         |                                                   |
|  Projection -> Spatial Tokens [B, 64, D]                    |
|         |                                                   |
|  +---------------------------------------------+            |
|  | Concat: [Spatial Tokens] + [Joint Queries]              |
|  |         [B, 64, D]      + [B, 16, D]    |                |
|  |              |                          |                |
|  |         [B, 80, D]                      |                |
|  +---------------------------------------------+            |
|         |                                                   |
|  +---------------------------------------------+            |
|  |     Self-Attention (4 layers)           |                |
|  |     - Joint tokens <-> Spatial tokens     |                |
|  |     - Bidirectional information exchange   |                |
|  +---------------------------------------------+            |
|         |                                                   |
|  Joint Tokens [B, 16, D] (latent representation)           |
|         |                                                   |
|         +---------------------------+                       |
|         |                       |                           |
|         |              +--------------------+               |
|         |              |  Heatmap Decoder   |               |
|         |              |  (Per-Joint)       |               |
|         |              +--------------------+               |
|         |                       |                           |
|         |              Recon Heatmap [16, 47, 47]           |
|         |              (loss: injects 2D info into tokens)  |
|         |                                                   |
|  +---------------------------------------------+            |
|  |     HMD Cross-Attention                 |                |
|  |     (adds 3D depth reference)           |                |
|  +---------------------------------------------+            |
|         |                                                   |
|  +------------------------------+                           |
|  |     3D Pose Head             |                           |
|  |     Linear(D, 3)             |                           |
|  +------------------------------+                           |
|         |                                                   |
|    3D Pose [B, 16, 3]                                       |
|                                                             |
+-------------------------------------------------------------+
```

### Information Flow

```
1. Self-Attention: Bidirectional interaction between Spatial <-> Joint
2. Heatmap Reconstruction: Forcefully injects 2D joint position information into tokens
3. HMD Cross-Attention: Adds 3D depth reference (head/hand positions)
4. 3D Pose Head: Predicts final 3D pose with rich 2D+3D information
```

---

## 4. Changes Compared to Existing Approach

| Item | Existing Attention Lifting | ViT-Style Lifting v2 |
|------|------------------------|----------------------|
| Query generation | soft_argmax(heatmap) -> embed | **Learnable parameters** |
| Attention | Cross-Attention | **Self-Attention + HMD Cross-Attn** |
| Information flow | Backbone -> Joint (unidirectional) | **Backbone <-> Joint (bidirectional)** |
| Heatmap role | soft_argmax input | **Reconstruction regularization** |
| 3D prediction | 2D + Depth concat | **Direct 3D Head** |
| 2D information injection | soft_argmax coordinates | **Reconstruction loss** |

---

## 5. Learnable Joint Queries

### Concept

```python
# A learnable query vector for each joint
self.joint_queries = nn.Parameter(torch.randn(1, 16, D))

# Instead of soft_argmax:
Q = self.joint_queries.expand(B, -1, -1)  # [B, 16, D]
```

### Advantages

1. **End-to-end learning**: No coordinate extraction bottleneck
2. **Rich representation**: Utilizes the full D-dimensional space (vs 2D coordinates)
3. **Proven approach**: Effectiveness demonstrated in DETR, TokenPose

### Learning Process

```
Initial: random initialization
After training: each joint query learns a "question" to find its corresponding joint

joint_queries[0] -> encodes the question "Where is the head?"
joint_queries[7] -> encodes the question "Where is the right wrist?"
```

---

## 6. Role of Self-Attention

### Cross-Attention vs Self-Attention

```
Cross-Attention (existing):
  Q: Joint tokens
  K, V: Spatial tokens
  -> Joint references Spatial (unidirectional)

Self-Attention (proposed):
  Q, K, V: [Spatial; Joint] all tokens
  -> Spatial <-> Joint bidirectional + Spatial <-> Spatial + Joint <-> Joint
```

### Benefits of Bidirectional

```
Joint -> Spatial: "Is there a joint at this position?" (obtaining depth information)
Spatial -> Joint: "Update features reflecting this joint information"
Joint -> Joint: "Learning relationships with other joints" (skeleton prior)
```

---

## 7. Expected Benefits

1. **Reduced information loss**: Eliminates soft_argmax coordinate extraction bottleneck
2. **Rich Queries**: Learned embeddings are more expressive than coordinates
3. **Skeleton relationship learning**: Joint <-> Joint self-attention
4. **Forced 2D information injection**: Reconstruction loss ensures joint position information in tokens
5. **Clear information separation**: 2D (reconstruction) -> 3D (HMD cross-attn)

---

## 8. Version-by-Version Structure Comparison

### v1 (Existing - Separate Heatmap Path)
```
Backbone -+-> Deconv -> Heatmap -> Heatmap Loss (separate)
          |
          +-> Spatial Tokens + Joint Queries
                      |
              Self-Attention
                      |
              Joint Tokens
                      |
              HMD Cross-Attention
                      |
              3D Pose Head -> 3D Loss
```
- Heatmap and 3D Lifting are **independent** (no information sharing)

---

### v2 (No Reconstruction)
```
Backbone -> Spatial Tokens + Joint Queries
                    |
            Self-Attention (spatial info automatically transferred)
                    |
            Joint Tokens
                    |
            HMD Cross-Attention
                    |
            3D Pose Head -> 3D Loss only
```
- No Heatmap
- Assumes Self-Attention naturally transfers spatial information

---

### v3 (Reconstruction Regularization) ⭐ Recommended
```
Backbone -> Spatial Tokens + Joint Queries
                    |
            Self-Attention
                    |
            Joint Tokens <-- shared latent
                    |
       +------------+------------+
       |                         |
       v                         v
Heatmap Decoder            HMD Cross-Attention
       |                         |
       v                         v
Recon Heatmap              3D Pose Head
       |                         |
       v                         v
Heatmap Loss <---------------> 3D Loss
(regularizes tokens)
```
- Heatmap Reconstruction **regularizes Joint Tokens**
- 2D information forcefully injected into joint tokens -> better 3D prediction

---

### Version Comparison Table

| | v1 | v2 | v3 |
|---|-----|-----|-----|
| Heatmap path | Separate (Deconv) | None | **Shared (Decoder)** |
| Heatmap role | Auxiliary only | - | **Token regularization** |
| Information flow | Independent | Self-Attn only | **Recon + Self-Attn** |
| Config | `use_heatmap_recon=True` + deconv | `use_heatmap_recon=False` | `use_heatmap_recon=True` |
| Complexity | High | Low | Medium |

### Smoke Test Results (1 epoch)

| | v2 | v3 |
|---|-----|-----|
| Full Body MPJPE | 303.59mm | **290.31mm** ✓ |
| Upper Body MPJPE | 206.22mm | **188.41mm** ✓ |
| Memory | 5891 MB | 7227 MB |

**Conclusion**: v3 recommended (Heatmap reconstruction improves joint token quality)

---

## 9. How to Run

```bash
# v2 (No Reconstruction)
python tools/train.py my_code/custom_config/HMD_xregopose_vit_lifting_v2_full_config.py

# v3 (With Reconstruction) ⭐ Recommended
python tools/train.py my_code/custom_config/HMD_xregopose_vit_lifting_v3_full_config.py
```

---

## 10. v3 Full Architecture Diagram (Detailed)

```
+-----------------------------------------------------------------------------+
|                           INPUT IMAGE [B, 3, 256, 256]                      |
+-----------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------+
|                         BACKBONE (ResNet-101)                               |
|  +----------+   +----------+   +----------+   +----------+                  |
|  |  layer1  | -> |  layer2  | -> |  layer3  | -> |  layer4  |                  |
|  | 256ch    |   | 512ch    |   | 1024ch   |   | 2048ch   |                  |
|  | 64x64    |   | 32x32    |   | 16x16    |   | 8x8      |                  |
|  +----------+   +----------+   +----------+   +----------+                  |
|                                                     |                        |
|                                     backbone_feat [B, 2048, 8, 8]           |
+-----------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------+
|                      SPATIAL PROJECTION                                      |
|                                                                              |
|  backbone_feat [B, 2048, 8, 8]                                              |
|         |                                                                    |
|         v                                                                    |
|  +------------------------------+                                           |
|  |  Conv2d(2048 -> 256, 1x1)    |  <- Channel dimension compression (2048 -> 256) |
|  |  BatchNorm2d(256)           |                                           |
|  |  ReLU                        |                                           |
|  +------------------------------+                                           |
|         |                                                                    |
|         v                                                                    |
|  spatial_feat [B, 256, 8, 8]                                                |
|         |                                                                    |
|         v  flatten(2) + transpose(1,2)                                      |
|  spatial_tokens [B, 64, 256]   <- 8x8=64 spatial tokens                    |
|         |                                                                    |
|         v  + PositionalEncoding2D (sinusoidal x,y)                          |
|         v  + spatial_type_embed (learnable)                                 |
|  spatial_tokens [B, 64, 256] (with position info)                           |
+-----------------------------------------------------------------------------+
                                        |
                                        |
          +-----------------------------+-----------------------------+
          |                                                           |
          v                                                           v
+---------------------------+                         +---------------------------+
|   SPATIAL TOKENS          |                         |   LEARNABLE JOINT QUERIES |
|   [B, 64, 256]            |                         |                           |
|                           |                         |  nn.Parameter             |
|   8x8 grid positions      |                         |  [1, 16, 256]             |
|   with pos encoding       |                         |         |                 |
|                           |                         |         v expand to B     |
|                           |                         |  + joint_type_embed       |
|                           |                         |  [B, 16, 256]             |
+---------------------------+                         +---------------------------+
          |                                                           |
          +-----------------------+-----------------------------------+
                                  | concat
                                  v
+-----------------------------------------------------------------------------+
|                         CONCATENATED TOKENS                                  |
|                         [B, 80, 256]                                         |
|                                                                              |
|         [Spatial: 64 tokens] + [Joint: 16 tokens]                           |
+-----------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                    SELF-ATTENTION LAYERS (x4)                                |
|  +------------------------------------------------------------------------+ |
|  |  TransformerEncoderLayer (Pre-LayerNorm)                               | |
|  |                                                                        | |
|  |       +------------------------------------------------------+        | |
|  |       |  LayerNorm                                       |             | |
|  |       |       |                                          |             | |
|  |       |  MultiheadAttention(D=256, heads=8)              |             | |
|  |       |  Q, K, V = all 80 tokens (bidirectional!)        |             | |
|  |       |       |                                          |             | |
|  |       |  Dropout + Residual                              |             | |
|  |       +------------------------------------------------------+        | |
|  |                           |                                            | |
|  |       +------------------------------------------------------+        | |
|  |       |  LayerNorm                                       |             | |
|  |       |       |                                          |             | |
|  |       |  MLP: Linear(256->1024) -> GELU -> Linear(1024->256) |             | |
|  |       |       |                                          |             | |
|  |       |  Dropout + Residual                              |             | |
|  |       +------------------------------------------------------+        | |
|  +------------------------------------------------------------------------+ |
|                           x 4 layers                                         |
|                                                                              |
|  ★ Key: Bidirectional interaction between Spatial <-> Joint                  |
|     - Joint -> Spatial: "Is there a joint at this position?"                 |
|     - Spatial -> Joint: "Update this joint's features"                       |
|     - Joint <-> Joint: "Relationships with other joints" (skeleton prior)    |
+-----------------------------------------------------------------------------+
                                  |
                                  v
                         LayerNorm(tokens)
                                  |
                                  v
                    Extract last 16 tokens (joint tokens)
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                         JOINT TOKENS [B, 16, 256]                            |
|                                                                              |
|  Each token = latent representation of the corresponding joint               |
|  Absorbed spatial information through Self-Attention                         |
+-----------------------------------------------------------------------------+
                    |                                     |
                    |                                     |
    +---------------+                                     +---------------+
    |                                                                     |
    v                                                                     v
+-------------------------------------+       +-------------------------------------+
|   HEATMAP DECODER (Reconstruction)  |       |                                     |
|   PerJointHeatmapDecoder            |       |   ★ 3D Prediction Path              |
|                                     |       |                                     |
|   joint_tokens [B, 16, 256]         |       +-------------------------------------+
|         |                           |                         |
|         v reshape to [B*16, 256]    |                         |
|   +-------------------------+       |                         v
|   |  MLP (shared)           |       |       +-------------------------------------+
|   |  Linear(256->256)        |       |       |         HMD CROSS-ATTENTION         |
|   |  LayerNorm + GELU       |       |       |                                     |
|   |  Linear(256->256)        |       |       |  Input:                             |
|   |  LayerNorm + GELU       |       |       |    Q: joint_tokens [B, 16, 256]     |
|   +-------------------------+       |       |    HMD_info [B, 9]                  |
|         |                           |       |         |                           |
|         v reshape [B*16, 256, 1, 1] |       |         v                           |
|   +-------------------------+       |       |  +-------------------------+        |
|   |  Progressive Upsample   |       |       |  |  HMD Embedding          |        |
|   |  1x1 -> 3x3 -> 6x6 ->      |       |       |  |  Linear(9->512->768)     |        |
|   |  12x12 -> 24x24 -> 47x47  |       |       |  |  reshape to [B, 3, 256] |        |
|   |                         |       |       |  |  (head, R_hand, L_hand) |        |
|   |  ConvTranspose2d x4     |       |       |  +-------------------------+        |
|   |  + Bilinear Upsample    |       |       |         |                           |
|   |  + Conv2d (final)       |       |       |         v hmd_tokens                |
|   +-------------------------+       |       |  +-------------------------+        |
|         |                           |       |  |  Cross-Attention        |        |
|         v                           |       |  |  (Pre-LayerNorm)        |        |
|   recon_heatmaps [B, 16, 47, 47]    |       |  |                         |        |
|         |                           |       |  |  Q: joint (normalized)  |        |
|         v                           |       |  |  K: hmd_tokens          |        |
|   +-------------------------+       |       |  |  V: hmd_tokens          |        |
|   |  Heatmap Recon Loss     |       |       |  |                         |        |
|   |  (vs GT heatmaps)       |       |       |  |  + Dropout + Residual   |        |
|   |                         |       |       |  +-------------------------+        |
|   |  ★ Forcefully injects   |       |       |         |                           |
|   |    2D joint position    |       |       |         v                           |
|   |    info into tokens!    |       |       |   refined_joints [B, 16, 256]       |
|   +-------------------------+       |       |                                     |
|         |                           |       |  ★ Adds 3D depth reference          |
|         v                           |       |    (using head/both hands positions)|
|   loss_heatmap_recon                |       +-------------------------------------+
|   (weight=500)                      |                         |
+-------------------------------------+                         |
                                                                v
                                      +---------------------------------------------+
                                      |              3D POSE HEAD                    |
                                      |                                              |
                                      |  refined_joints [B, 16, 256]                |
                                      |         |                                    |
                                      |         v                                    |
                                      |  +--------------------------+               |
                                      |  |  Linear(256 -> 256)       |               |
                                      |  |  LayerNorm(256)          |               |
                                      |  |  GELU                    |               |
                                      |  |  Linear(256 -> 128)       |               |
                                      |  |  GELU                    |               |
                                      |  |  Linear(128 -> 3)         |               |
                                      |  +--------------------------+               |
                                      |         |                                    |
                                      |         v                                    |
                                      |   pose_3d [B, 16, 3]                        |
                                      |   (x, y, z for each joint)                  |
                                      +---------------------------------------------+
                                                                |
                                                                v
+---------------------------------------------------------------------------------+
|                                    LOSSES                                        |
|                                                                                  |
|  +----------------------------------------------------------------------------+ |
|  |  3D Pose Losses:                                                           | |
|  |    - loss_pose_l2norm     (weight=1.0)   - L2 distance                    | |
|  |    - loss_cosine_similarity (weight=0.1) - Direction consistency          | |
|  |    - loss_limb_length     (weight=0.25)  - Bone length consistency        | |
|  +----------------------------------------------------------------------------+ |
|                                                                                  |
|  +----------------------------------------------------------------------------+ |
|  |  HMD Loss:                                                                 | |
|  |    - loss_hmd (weight=1.0) - Reconstructed HMD vs GT HMD                  | |
|  |      pose_3d -> _compute_hmd_from_pose() -> hmd_recon [B, 9]              | |
|  +----------------------------------------------------------------------------+ |
|                                                                                  |
|  +----------------------------------------------------------------------------+ |
|  |  Heatmap Reconstruction Loss (v3 core!):                                   | |
|  |    - loss_heatmap_recon (weight=500)                                      | |
|  |    - recon_heatmaps vs gt_heatmaps                                        | |
|  |    ★ Forces joint tokens to encode 2D joint positions                     | |
|  +----------------------------------------------------------------------------+ |
+---------------------------------------------------------------------------------+
                                                                |
                                                                v
+---------------------------------------------------------------------------------+
|                                OUTPUT                                            |
|                                                                                  |
|   pose_3d [B, 16, 3]  ->  3D coordinates (x, y, z) of 16 joints                |
|                                                                                  |
|   Joint order:                                                                   |
|   [0] Head        [4] L_Wrist     [8] R_Knee      [12] L_Knee                   |
|   [1] Neck        [5] L_Elbow     [9] R_Ankle     [13] L_Ankle                  |
|   [2] R_Shoulder  [6] L_Shoulder  [10] R_Foot     [14] L_Foot                   |
|   [3] R_Elbow     [7] R_Wrist     [11] L_Hip      [15] Pelvis                   |
+---------------------------------------------------------------------------------+
```

### v3 Information Flow Summary

```
1. Backbone -> Spatial Tokens [64, D]
2. + Learnable Joint Queries [16, D]
3. Self-Attention (bidirectional interaction)
4. Joint Tokens [16, D] <- shared latent representation
       |
       +-->  Heatmap Decoder -> Recon Loss (2D information injection)
       |
       +-->  HMD Cross-Attention (3D depth reference)
                   |
                   +-->  3D Pose Head -> [16, 3]
```

### Structural Considerations

**Heatmap Decoder Position:**
- Currently: Positioned **before** HMD Cross-Attention
- Heatmap is a pure visual task -> should work without HMD
- Therefore the current position is appropriate

**Gradient Flow:**
```
loss_heatmap_recon -> heatmap_decoder -> joint_tokens (before HMD)
                                              |
                                    Self-Attention layers

loss_3d -> head_3d -> joint_tokens (after HMD) -> hmd_cross_attn
                                              |
                                    Self-Attention layers
```
- Both losses propagate to the backbone through Self-Attention

---

## 11. v3 Training Stability Analysis (Validation Spike Cause)

### Experimental Results (10 epochs)

```
Epoch | Full Body | Upper Body | Notes
------|-----------|------------|------
  1   |   50.32   |   31.40    |
  2   |   53.72 ^ |   28.27 v  | Full ^, Upper v (inconsistency)
  3   |   48.28 v |   25.46 v  | LR decay (0.5x)
  4   |   45.34 v |   23.49 v  | ★ Best
  5   |   47.13 ^ |   24.76 ^  | LR decay -> spike!
  6   |   49.12 ^ |   26.22 ^  | Spike continues
  7   |   45.59 v |   24.53 v  | LR decay -> recovery
  8   |   47.11 ^ |   25.49 ^  | Spike again
  9   |   47.81   |   26.01    |
 10   |   47.37   |   24.56    |
```

### Extreme Fluctuation in Walking Action

```
Epoch | Walking MPJPE | Change
------|---------------|------
  1   |    52.38      |
  2   |    27.80      | vv
  3   |    76.97      | ^^^
  4   |    59.62      |
  5   |    25.28      | vv
  6   |    93.95      | ^^^ (worst)
  7   |    54.86      |
  8   |    22.99      | vv (best)
  9   |    86.72      | ^^^
 10   |    50.51      |
```
- **22mm ~ 94mm** range with extreme fluctuation
- Particularly unstable for actions with significant lower body movement

---

### Spike Cause Analysis

#### 1. LR Schedule Issue (MultiStepLR)

```python
# Current settings
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5),
]
```

**LR Changes:**
```
Epoch 1-3: LR = 0.0005
Epoch 4:   LR = 0.00025 (decay at 3) -> Best!
Epoch 5:   LR = 0.000125 (decay at 5) -> Spike begins
Epoch 7:   LR = 0.0000625 (decay at 7)
```

**Problem**: Abrupt LR reduction causes model instability

---

#### 2. Limitations of HMD Information

```
Information provided by HMD:
  - Head position ✓
  - Right hand position ✓
  - Left hand position ✓

Not provided by HMD:
  - Lower body (legs, feet) ✗
```

**Result:**
- Upper Body: Stable prediction with HMD information
- Lower Body: Relies purely on visual features -> unstable
- Lower body prediction fails for dynamic actions like walking

---

#### 3. Heatmap Reconstruction Loss Weight

```python
loss_heatmap_recon=dict(loss_weight=500, ...)  # Very high compared to other losses
```

| Loss | Weight | Ratio |
|------|--------|------|
| loss_pose_l2norm | 1.0 | 1x |
| loss_cosine_similarity | 0.1 | 0.1x |
| loss_limb_length | 0.25 | 0.25x |
| loss_hmd | 1.0 | 1x |
| **loss_heatmap_recon** | **500** | **500x** |

**Problem:**
- Reconstruction quality fluctuation -> large total loss fluctuation
- 2D heatmap learning becomes dominant -> may interfere with 3D prediction

---

#### 4. PerJointHeatmapDecoder Structure

```python
# Extreme upsampling from 1x1 -> 47x47 (2209x)
self.upsample = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 3, 1, 0),  # 1->3
    nn.BatchNorm2d(128),  # <- Unstable at small spatial sizes
    nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 3->6
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 6->12
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 64, 4, 2, 1),    # 12->24
    nn.BatchNorm2d(64),
    nn.Upsample(size=(47, 47)),              # 24->47
    ...
)
```

**Problems:**
- Extreme upsampling starting from 1x1
- BatchNorm statistics unstable at small spatial sizes
- ConvTranspose checkerboard artifacts

---

#### 5. Full Body vs Upper Body Inconsistency

Epoch 2 example:
```
Full Body:  50.32 -> 53.72 (worsened)
Upper Body: 31.40 -> 28.27 (improved)
```

**Implication:**
- Hands/arms improve while legs worsen
- HMD Cross-Attention is effective only for upper body

---

### Solutions

#### Option 1: Modify LR Schedule (Recommended)

```python
# CosineAnnealingLR for smooth decay
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=1e-6,
        by_epoch=True,
    ),
]
```

#### Option 2: Reduce Heatmap Recon Loss Weight

```python
# Reduce from 500 -> 100~200
loss_heatmap_recon=dict(loss_weight=200, ...)
```

#### Option 3: Increase Lower Body Weight

```python
# Higher weight for lower body joints
keypoint_weights = [
    1.0,  # Head
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Upper body
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  # Lower body (1.5x)
]
```

#### Option 4: BatchNorm -> GroupNorm

```python
# In PerJointHeatmapDecoder
nn.BatchNorm2d(128)  # Unstable
    |
nn.GroupNorm(8, 128)  # More stable
```

---

### Recommended Experiment Order

1. **Option 1 (LR Schedule)** first - Simplest, expected to have the most impact
2. If ineffective, add **Option 2 (Loss Weight)**
3. If still unstable, apply **Option 4 (GroupNorm)**

---

## 12. v4: CosineAnnealingLR Applied

### Changes from v3

```python
# v3 (MultiStepLR - abrupt decay)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5),
]

# v4 (CosineAnnealingLR - smooth decay)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=1e-6,
        by_epoch=True,
    ),
]
```

### LR Change Comparison

```
v3 MultiStepLR:
  Epoch 1-3: 0.0005
  Epoch 4:   0.00025 (sudden 50% decrease!)
  Epoch 5:   0.000125 (sudden 50% decrease!)
  -> Abrupt changes cause validation spikes

v4 CosineAnnealingLR:
  Epoch 0:  0.0005
  Epoch 5:  ~0.00025 (smooth decrease)
  Epoch 10: 0.000001
  -> Smooth changes for stable training
```

### Smoke Test Results (2 epochs)

| Version | Epoch 1 Full Body | Epoch 2 Full Body |
|---------|-------------------|-------------------|
| v3 (MultiStepLR) | 290.31mm | - |
| **v4 (CosineAnnealingLR)** | **281.73mm** | - |

**Conclusion**: v4 shows more stable convergence from the start

---

## 13. v5: Hybrid Attention Structure

### Key Changes from v4

1. **Attention structure change**
   - v4: Self-Attention [80x80] (Spatial 64 + Joint 16)
   - v5: Cross-Attention [16x64] + Self-Attention [16x16]

2. **Role separation**
   - Cross-Attention: Joints find their locations in the image
   - Self-Attention: Learning skeleton relationships between joints

3. **Computational efficiency**
   - 6400 attention weights -> 1280 (5x reduction)

4. **Gradient Scaling**
   - Heatmap gradient scaled by 0.1x
   - Focuses on 3D prediction learning

### v5 Architecture Diagram

```
+-------------------------------------------------------------+
|  Backbone feat [2048, 8, 8]                                 |
|         |                                                   |
|  Spatial Tokens [64, D]                                     |
|         |                                                   |
|  +---------------------------------------------+            |
|  |  Stage 1: Cross-Attention (J -> S)       |                |
|  |  Q: Joint Queries [16, D]               |                |
|  |  K/V: Spatial Tokens [64, D]            |  [16x64]       |
|  |  -> Each joint finds its location in the image |          |
|  +---------------------------------------------+            |
|         |                                                   |
|  +---------------------------------------------+            |
|  |  Stage 2: Self-Attention (J -> J) x N    |                |
|  |  Q=K=V: Joint Tokens [16, D]            |  [16x16]       |
|  |  -> Learning skeleton relationships between joints |      |
|  +---------------------------------------------+            |
|         |                                                   |
|  Joint Tokens [16, D]                                       |
|         |                                                   |
|         +-- Heatmap Decoder (gradient scaled by 0.1)        |
|         |                                                   |
|  +---------------------------------------------+            |
|  |  Stage 3: HMD Cross-Attention           |                |
|  |  Q: Joint Tokens, K/V: HMD Tokens       |  [16x3]        |
|  |  -> Adds 3D depth reference              |                |
|  +---------------------------------------------+            |
|         |                                                   |
|    3D Pose Head -> [16, 3]                                   |
+-------------------------------------------------------------+
```

### v4 vs v5 Attention Comparison

```
v4 Self-Attention [80x80]:
+-------------------------------------------------------------+
|                    Q = K = V (80 tokens)                     |
|  +--------------------------------------------------------+ |
|  |  S->S (64x64)  |  S->J (64x16)  |  <- Unnecessary computation |
|  |  J->S (16x64)  |  J->J (16x16)  |  <- Necessary computation   |
|  +--------------------------------------------------------+ |
|  Total: 6400 attention weights                              |
|  Problem: S->S, S->J are unnecessary for 3D prediction     |
+-------------------------------------------------------------+

v5 Hybrid Attention:
+-------------------------------------------------------------+
|  Stage 1: Cross-Attention (J -> S) [16x64]                   |
|  +--------------------------------------------------------+ |
|  |  Q: Joint [16, D]                                      | |
|  |  K, V: Spatial [64, D]                                 | |
|  |  -> Each joint attends to relevant spatial positions    | |
|  +--------------------------------------------------------+ |
|                                                             |
|  Stage 2: Self-Attention (J -> J) [16x16]                    |
|  +--------------------------------------------------------+ |
|  |  Q = K = V: Joint [16, D]                              | |
|  |  -> Learning skeleton relationships between joints     | |
|  +--------------------------------------------------------+ |
|                                                             |
|  Total: 1024 + 256 = 1280 attention weights                 |
|  Advantage: Focuses only on necessary attention             |
+-------------------------------------------------------------+
```

### Gradient Scaling Implementation

```python
class GradientScale(torch.autograd.Function):
    """Forward is normal, backward is scaled."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

# Usage
if self.use_heatmap_recon:
    # Heatmap gradient = 0.1x (focus on 3D prediction learning)
    scaled_tokens = gradient_scale(joint_tokens, 0.1)
    recon_heatmaps = self.heatmap_decoder(scaled_tokens)
```

### Smoke Test Results (2 epochs)

| Version | Epoch 1 Full Body | Epoch 2 Full Body |
|---------|-------------------|-------------------|
| v4 (Self-Attn) | 290.31mm | 281.73mm |
| **v5 (Hybrid)** | **272.93mm** | **204.41mm** |

**Conclusion**: v5 shows faster convergence

---

## 14. Methods to Inject Depth Information into Joint Tokens

### Current Problem

```
Joint Queries (learnable)  ->  Learning 3D with only 2D spatial attention
                              |
                          Depth cues are only transferred implicitly
```

### Proposed Solutions

#### Option 1: Depth-Aware Joint Initialization

```python
# Existing: random initialization
joint_queries = Parameter(randn(1, 16, D))

# Improved: include skeleton depth prior
joint_queries = Parameter(
    concat([
        learnable_2d,      # [16, D//2] - learn 2D position
        depth_prior        # [16, D//2] - skeleton-based depth prior
    ])
)
```

- Encoding the average depth order of the human skeleton
  - head -> shoulder -> elbow -> wrist (sequential)
- Providing typical depth range per joint as a prior

#### Option 2: Multi-Scale Spatial Features

```
Current: backbone feat [8x8] -> spatial tokens [64]

Improved: multi-scale features
    - [8x8]   -> 64 tokens (coarse, depth context)
    - [16x16] -> 256 tokens (fine, location)
    Cross-Attention: Joint queries -> Multi-scale tokens
```

- Small feature map: wide receptive field -> contains depth context
- Large feature map: precise localization

#### Option 3: Early HMD Injection (Recommended)

```
Current:
    Cross-Attn (J->S) -> Self-Attn (J->J) -> HMD Cross-Attn (at the very end)

Improved:
    Cross-Attn (J->S) -> [HMD Injection] -> Self-Attn (J->J)
                              ^
                        Provides depth anchor early on
```

- Inject head/hand depth provided by HMD at the early stage
- Other joints learn relative depth based on these anchors
- **Simplest to implement, maximizes HMD utilization**

#### Option 4: Explicit Depth Token

```python
# Joint Tokens [16, D] + Depth Token [1, D]

depth_token = nn.Parameter(torch.randn(1, 1, D))  # Global depth context

# In Self-Attention
tokens = concat([joint_tokens, depth_token])  # [17, D]
tokens = self_attention(tokens)
joint_tokens = tokens[:, :16]  # Exclude depth token
```

- A separate token responsible for global depth context
- All joints attend to this token
- Serves as depth reference in Self-Attn

### Recommended Implementation Priority

1. **Option 3 (Early HMD Injection)** - Simplest, planned for implementation in v6
2. **Option 1 (Depth-Aware Init)** - Leverages skeleton prior
3. **Option 2 (Multi-Scale)** - Increases computation but effective

---

## 15. Version Summary

| Version | Structure | LR Schedule | Attention | Features |
|---------|------|-------------|-----------|------|
| v1 | Separate Heatmap + Lifting | - | - | Heatmap/3D independent |
| v2 | Self-Attention only | MultiStep | Self [80x80] | No Recon |
| v3 | Self-Attention + Recon | MultiStep | Self [80x80] | Recon regularization |
| **v4** | Self-Attention + Recon | **CosineAnnealing** | Self [80x80] | Stable LR decay |
| **v5** | **Hybrid Attention** + Recon | CosineAnnealing | **Cross [16x64] + Self [16x16]** | Role separation, Grad Scale |
| v6 (planned) | Hybrid + Early HMD | CosineAnnealing | Cross + Self | Depth prior enhancement |

---

## 16. Future Extensions (TODO)

### Phase 3: High-Resolution Extension
- Use layer3 (8x8 -> 16x16)
- 256 spatial tokens

### v6: Early HMD Injection
- Move HMD Cross-Attention before Self-Attention
- Provide depth anchor early on

---

## 17. References

- **ViTPose**: Simple Vision Transformer Baselines for Human Pose Estimation (NeurIPS 2022)
- **TokenPose**: Learning Keypoint Tokens for Human Pose Estimation (ICCV 2021)
- **DETR**: End-to-End Object Detection with Transformers (ECCV 2020)
