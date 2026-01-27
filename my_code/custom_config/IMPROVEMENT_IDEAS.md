# EgoPose 3D Improvement Ideas

> Last updated: 2026-01-21

---

## Table of Contents

1. [EfficientHeatmapDecoder Improvement](#efficientheatmapdecoder-improvement)
2. [Dual Backbone Mutual Learning](#dual-backbone-mutual-learning)
3. [Structural Problem: Limitations of 3D Information Encoding in Heatmaps](#structural-problem-limitations-of-3d-information-encoding-in-heatmaps)
4. [Backbone Feature Fusion](#backbone-feature-fusion)
5. [Attention-based Lifting Network](#attention-based-lifting-network)
6. [Cross Attention-based HMD Fusion](#cross-attention-based-hmd-fusion)

---

## EfficientHeatmapDecoder Improvement

### Current Architecture Analysis

```
Z [64] → FC → [256] → reshape → [256, 1, 1]
              ↓
      ConvTranspose2d (5 stages)
              ↓
         [64, 32, 32]
              ↓
    Bilinear Upsample + Conv
              ↓
      Heatmap [16, 47, 47]
```

**Current parameters**: 1.35M (96.6% reduction compared to original 40M)

**Problems**:
1. Z[64] is processed only by FC → lack of guidance for spatial structure
2. Simple sequential upsampling → information flow is unidirectional
3. All joints are generated through the same path → difficult to reflect joint-specific characteristics

---

### Improvement Option 1: AdaIN (Adaptive Instance Normalization) ⭐ Recommended

A method where Z is used as a style to influence each layer (StyleGAN style):

```python
class AdaINHeatmapDecoder(nn.Module):
    """StyleGAN approach: Z influences the normalization of each layer"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        # Learned constant (starting point) - 4x4 spatial
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4))

        # Z → style parameters (scale, shift for each layer)
        self.style_fc = nn.ModuleList([
            nn.Linear(input_size, 256 * 2),  # layer 1: scale + shift
            nn.Linear(input_size, 128 * 2),  # layer 2
            nn.Linear(input_size, 64 * 2),   # layer 3
        ])

        # Upsampling layers
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)  # 4x4 → 8x8
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)   # 8x8 → 16x16
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)    # 16x16 → 32x32
        self.final = nn.Conv2d(32, num_classes, 1)      # → 47x47

    def adain(self, feat, style):
        """Adaptive Instance Normalization"""
        B, C, H, W = feat.shape
        feat = F.instance_norm(feat)
        scale, shift = style.view(B, 2, C).chunk(2, dim=1)
        return feat * scale.squeeze(1).view(B, C, 1, 1) + shift.squeeze(1).view(B, C, 1, 1)

    def forward(self, z):
        B = z.size(0)
        x = self.const.expand(B, -1, -1, -1)  # [B, 256, 4, 4]

        # Stage 1: 4x4 → 8x8
        x = self.adain(x, self.style_fc[0](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv1(x), 0.2)

        # Stage 2: 8x8 → 16x16
        x = self.adain(x, self.style_fc[1](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv2(x), 0.2)

        # Stage 3: 16x16 → 32x32
        x = self.adain(x, self.style_fc[2](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv3(x), 0.2)

        # Final: 32x32 → 47x47
        x = F.interpolate(x, size=(47, 47), mode='bilinear', align_corners=False)
        heatmap = self.final(x)

        return heatmap
```

**Advantages**:
- Z **continuously influences** the entire generation process
- Starts from learned constant → more stable training
- Minimal parameter increase (~0.1M)

**Expected parameters**: ~1.5M

---

### Improvement Option 2: PixelShuffle-based Upsampling

Using PixelShuffle instead of ConvTranspose2d (reduces checkerboard artifacts):

```python
class PixelShuffleHeatmapDecoder(nn.Module):
    """Eliminates checkerboard artifacts with PixelShuffle"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        self.fc = nn.Linear(input_size, 256 * 4 * 4)  # Directly to 4x4

        # PixelShuffle: Conv → channel shuffle → spatial increase
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 256 * 4, 3, padding=1),  # 4x channels for 2x upscale
            nn.PixelShuffle(2),  # [256*4, H, W] → [256, 2H, 2W]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )  # 4x4 → 8x8

        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )  # 8x8 → 16x16

        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )  # 16x16 → 32x32

        self.final = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        x = self.up1(x)   # [B, 256, 8, 8]
        x = self.up2(x)   # [B, 128, 16, 16]
        x = self.up3(x)   # [B, 64, 32, 32]
        x = self.final(x) # [B, 16, 47, 47]
        return x
```

**Advantages**:
- Eliminates checkerboard artifacts
- Smoother upsampling
- Similar parameters to ConvTranspose2d

---

### Improvement Option 3: Joint-wise Parallel Generation

Independent heatmap generation for each joint:

```python
class JointWiseHeatmapDecoder(nn.Module):
    """Each joint is generated through an independent decoder path"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        # Z → separate latent per joint
        self.joint_fc = nn.Linear(input_size, num_classes * 32)  # [B, 16*32]

        # Shared upsampler (parameter efficiency)
        self.shared_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 1→2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 2→4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 16→32
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Per-joint refinement (different weights for each)
        self.joint_refine = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
                nn.Conv2d(16, 1, 3, padding=1)
            ) for _ in range(num_classes)
        ])

    def forward(self, z):
        B = z.size(0)
        # Z → separate per joint
        joint_z = self.joint_fc(z).view(B, 16, 32)  # [B, 16, 32]

        heatmaps = []
        for i in range(16):
            jz = joint_z[:, i, :].view(B, 32, 1, 1)  # [B, 32, 1, 1]
            feat = self.shared_upsample(jz)          # [B, 16, 32, 32]
            hm = self.joint_refine[i](feat)          # [B, 1, 47, 47]
            heatmaps.append(hm)

        return torch.cat(heatmaps, dim=1)  # [B, 16, 47, 47]
```

**Advantages**:
- Individually learns characteristics of each joint (size, distribution)
- Shared backbone + per-joint head architecture

**Disadvantages**:
- Potential speed degradation due to sequential processing (parallelization needed)

---

### Improvement Option 4: Attention-based Upsampling

Utilizing global context with self-attention:

```python
class AttentionHeatmapDecoder(nn.Module):
    """Global context via self-attention at low resolution"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        self.fc = nn.Linear(input_size, 256 * 4 * 4)  # Directly to 4x4

        # Self-attention at 4x4 (16 tokens - efficient)
        self.self_attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(256)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16→32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        B, C, H, W = x.shape

        # Self-attention on 4x4 = 16 tokens
        x_flat = x.flatten(2).transpose(1, 2)  # [B, 16, 256]
        x_attn, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_attn = self.attn_norm(x_flat + x_attn)  # Residual
        x = x_attn.transpose(1, 2).view(B, C, H, W)

        x = self.upsample(x)
        heatmap = self.final(x)
        return heatmap
```

**Advantages**:
- Models inter-joint relationships through attention
- Attention at low resolution (4x4) → efficient

---

### Improvement Option 5: UNet-style Skip Connection

Connecting encoder's intermediate features to the decoder:

```
Current Encoder architecture:
Heatmap [16, 47, 47]
    ↓ conv1 (stride 2)
[64, 24, 24]  ← skip1
    ↓ conv2 (stride 2)
[128, 12, 12] ← skip2
    ↓ conv3 (stride 2)
[256, 6, 6]   ← skip3
    ↓ flatten + FC
Z [64]

Improved Decoder:
Z [64]
    ↓ FC + reshape
[256, 6, 6]
    ↓ ConvT + concat(skip3)
[128, 12, 12]
    ↓ ConvT + concat(skip2)
[64, 24, 24]
    ↓ ConvT + concat(skip1)
[32, 47, 47]
    ↓ Conv
Heatmap [16, 47, 47]
```

**Advantages**:
- Improved reconstruction accuracy through encoder information reuse
- Preserves low-level details

**Disadvantages**:
- Requires encoder modification (returning intermediate features)

---

### Improvement Option Comparison

| Option | Implementation Difficulty | Expected Effect | Parameter Increase | Recommendation Rank |
|--------|--------------------------|-----------------|-------------------|---------------------|
| **AdaIN** | Medium | ⭐⭐⭐ | ~0.1M | **1** |
| **PixelShuffle** | Low | ⭐⭐ | None | **2** |
| Attention (low-res) | Medium | ⭐⭐⭐ | ~0.2M | 3 |
| Joint-wise | High | ⭐⭐ | ~0.5M | 4 |
| UNet Skip | High | ⭐⭐⭐ | None | 5 |

**Recommendation**: AdaIN + PixelShuffle combination

---

### Implementation Priority

1. **AdaIN-based decoder** - Z influences the entire generation
2. **PixelShuffle** - eliminates checkerboard artifacts
3. Add Attention if needed

---

## Dual Backbone Mutual Learning

Knowledge Transfer strategy in a Dual Backbone architecture using different Pretrained Weights (COCO, MPII).

### Current Implementation Analysis

```python
# custom_egopose_baselinel1_head_multi_backbone.py
loss_backbone_latant = MSE(backbone_feat, backbone_feat2)
loss_backbone_heatmap = MSE(final_heatmap, final_heatmap2)
```

### Problems
1. **Diversity loss**: Trying to make two features identical
2. **Loss of pretrained strengths**: Unique knowledge from COCO/MPII is destroyed
3. **Lack of directionality**: Bidirectional gradient makes it unclear which side is the teacher

---

### 1. Ensemble Teacher (Recommended)

Creating a better pseudo-teacher by ensembling outputs from both backbones.

```
┌───────────────┐     ┌───────────────┐
│  COCO Backbone│     │  MPII Backbone│
│    (feat1)    │     │    (feat2)    │
└───────┬───────┘     └───────┬───────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────┐
│     Ensemble (Weighted Average)      │
│  feat_ensemble = w1*feat1 + w2*feat2 │
└───────────────────┬─────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   KD to feat1             KD to feat2
```

```python
def ensemble_mutual_learning_loss(feat1, feat2, heatmap1, heatmap2):
    # Confidence-based dynamic weighting
    conf1 = heatmap1.max(dim=-1)[0].max(dim=-1)[0].mean()
    conf2 = heatmap2.max(dim=-1)[0].max(dim=-1)[0].mean()

    w1 = conf1 / (conf1 + conf2 + 1e-6)
    w2 = conf2 / (conf1 + conf2 + 1e-6)

    # Ensemble feature (gradient blocked with detach)
    feat_ensemble = w1 * feat1.detach() + w2 * feat2.detach()

    # Each backbone mimics the ensemble
    loss1 = F.mse_loss(feat1, feat_ensemble)
    loss2 = F.mse_loss(feat2, feat_ensemble)

    return loss1 + loss2
```

---

### 2. Progressive Warmup

Preserving pretrained weights initially, gradually introducing mutual learning.

```python
def get_mutual_loss_weight(epoch, warmup_epochs=5, rampup_epochs=10):
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + rampup_epochs:
        return (epoch - warmup_epochs) / rampup_epochs
    else:
        return 1.0
```

---

### 3. Heatmap KL Divergence

Deep Mutual Learning paper approach:

```python
def heatmap_kl_divergence_loss(heatmap1, heatmap2, temperature=4.0):
    B, K, H, W = heatmap1.shape

    h1_flat = heatmap1.view(B, K, -1) / temperature
    h2_flat = heatmap2.view(B, K, -1) / temperature

    p1 = F.softmax(h1_flat, dim=-1)
    p2 = F.softmax(h2_flat, dim=-1)

    kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2 * (temperature ** 2)
```

---

### Dual Backbone Experiment Priority

| Rank | Method | Reason |
|------|--------|--------|
| 1 | Ensemble Teacher | Simple implementation, proven effectiveness |
| 2 | Progressive Warmup | Easy to add to existing approach |
| 3 | Heatmap KL Div | Reproducing the paper method |

---

## Structural Problem: Limitations of 3D Information Encoding in Heatmaps

### Core Problem

```
Backbone feat [2048, 8, 8]  ← Rich 3D information (texture, context, depth cues)
       ↓ (Deconv)
Heatmap [16, 47, 47]        ← Only 2D position information remains (depth lost!)
       ↓ (CNN Encoder)
Z [64]                      ← Extreme compression (additional loss)
       ↓
3D Pose                     ← Ambiguity due to insufficient depth information
```

### Academic Basis

1. **Depth Ambiguity Problem**: "A single 2D pose can map to multiple 3D poses"
2. **Information Loss in CNN Encoder**: "CNN-based encoder fails to properly preserve heatmap information"
3. **Lifting by Image**: "Rich semantic/texture information from images contributes to more accurate lifting"

### Conclusion

**Heatmaps are optimized for 2D position encoding** and have difficulty containing 3D depth information.
→ **Backbone features must be utilized separately** to preserve depth cues.

---

## Backbone Feature Fusion

### Option A: Backbone + Heatmap Latent Concat (Recommended)

```
Backbone feat [2048, 8, 8]
       │
       ├──→ GAP → [2048] → FC → [256]  ← Backbone latent
       │                        │
       ↓ (Deconv)               │
Heatmap [16, 47, 47]            │
       ↓ (Encoder)              │
Z_hm [64]                       │
       │                        │
       └──── Concat ────────────┘
              ↓
         [64 + 256 + 64(HMD)] = [384]
              ↓
         Pose Decoder → 3D Pose
```

### Option B: 2D Coords + Backbone Feature

```
Heatmap → soft-argmax → 2D coords [16, 2] + conf [16]
                              │
Backbone feat → GAP → FC → Context [256]
                              │
              Concat ─────────┘
                 ↓
           [32 + 16 + 256 + 9(HMD)] = 313
                 ↓
           Lifting Network → 3D Pose
```

---

## Attention-based Lifting Network

### Cross-Attention (2D → Backbone) ⭐ Recommended

Each joint's 2D position queries the corresponding depth information from the backbone features:

```
Query: 2D joint tokens [B, 16, D]    ← "What is the depth at this 2D position?"
Key/Value: Backbone tokens [B, 64, D] ← "Spatial depth information"
                    ↓
            Cross-Attention
                    ↓
      Depth-aware joint features [B, 16, D]
                    ↓
                3D Pose
```

```python
class CrossAttentionLifting(nn.Module):
    def __init__(self, joint_dim=64, backbone_dim=256, num_heads=4):
        super().__init__()

        self.joint_embed = nn.Linear(2, joint_dim)
        self.backbone_proj = nn.Conv2d(2048, backbone_dim, 1)
        self.kv_proj = nn.Linear(backbone_dim, joint_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads, batch_first=True)

        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, coords_2d, backbone_feat, confidence, hmd_info):
        # Joint queries from 2D coords
        joint_q = self.joint_embed(coords_2d)  # [B, 16, 64]

        # Backbone spatial tokens
        backbone_tokens = self.backbone_proj(backbone_feat).flatten(2).transpose(1, 2)
        backbone_kv = self.kv_proj(backbone_tokens)  # [B, 64, 64]

        # Cross attention
        depth_features, attn_weights = self.cross_attn(
            query=joint_q, key=backbone_kv, value=backbone_kv)

        pose_3d = self.output_proj(depth_features)
        return pose_3d, attn_weights
```

**Advantages**:
1. Selective depth querying: each joint gets depth information from the needed spatial location
2. Interpretable: visualize which positions were referenced via `attn_weights`
3. Role separation: 2D=Query (position), Backbone=Key/Value (depth)

---

## Cross Attention-based HMD Fusion

### Joint-wise Cross Attention (Recommended)

Each joint independently attends to HMD information:

```python
class JointHMDCrossAttention(nn.Module):
    def __init__(self, joint_dim=64, num_heads=4):
        super().__init__()

        # HMD as 3 tokens (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads, batch_first=True)

    def forward(self, joint_features, hmd_info):
        B = joint_features.size(0)
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)  # [B, 3, D]

        attn_out, _ = self.cross_attn(
            query=joint_features,  # [B, 16, D]
            key=hmd_tokens,        # [B, 3, D]
            value=hmd_tokens)

        return joint_features + attn_out  # Residual
```

**Effect**: Natural mapping of hand joints → hand HMD, torso → head HMD

---

## Hybrid Lifting: Baseline + Attention Refinement ⭐ NEW

> Added 2026-01-27

### Motivation

| Model | MPJPE | Characteristics |
|-------|-------|-----------------|
| **Baseline** | **41.37mm** 🏆 | Conv Encoder + Linear, stable |
| ViT Lifting v3 | 45.34mm | Full Attention, unstable training |

**Goal**: Combine Baseline stability + ViT's inter-joint relationship modeling

---

### Option Comparison

| Option | Change | Risk |
|--------|--------|------|
| 1. Attention HMD Fusion | Only change HMD fusion | Minimal |
| 2. Joint-wise Feature | Per-joint pooling | Medium |
| **3. Conv + Attention Refinement** | **Keep Conv Encoder + supplement with Attention** | **Low** |

---

### Option 3 Detailed Design (Selected)

```
Heatmap [16, 47, 47]
         ↓
┌─────────────────────────────────────────┐
│  Conv Encoder (same as Baseline)        │
│  Conv: 16→64→128→256, GAP → 64-dim      │
└─────────────────────────────────────────┘
         ↓
      Z [B, 64]
         ↓
┌─────────────────────────────────────────┐
│  Z Reshape: [B, 64] → [B, 16, 4]        │
│  (4-dim latent per joint)               │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Joint Embedding: [B, 16, 4] → [B, 16, D] │
│  Linear(4 → joint_dim)                  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Self-Attention (inter-joint relations) │
│  Query/Key/Value: [B, 16, D]            │
│  Output: [B, 16, D]                     │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  HMD Cross-Attention                    │
│  Query: Joint tokens [B, 16, D]         │
│  Key/Value: HMD tokens [B, 3, D]        │
│  (head, right_hand, left_hand)          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Output Head                            │
│  Linear(D → 3) per joint                │
│  Output: [B, 16, 3]                     │
└─────────────────────────────────────────┘
```

---

### Core Design Principles

1. **Conv Encoder reuse**: Proven heatmap→latent transformation from Baseline
2. **Z decomposition**: Reinterpret 64-dim as 16 joints x 4-dim
3. **Self-Attention**: Inter-joint relationships (skeleton structure)
4. **Cross-Attention HMD**: Selective reference to HMD information per joint
5. **LinearModel removal**: Replaced by Attention

---

### Implementation Code (CustomEgoposeHybridLiftingHead)

```python
class HybridLiftingModule(nn.Module):
    """Baseline Conv Encoder + Attention Refinement"""

    def __init__(self, num_joints=16, latent_dim=64, joint_dim=64,
                 num_heads=4, num_self_attn_layers=2, dropout=0.1):
        super().__init__()

        # Decompose Z per joint: 64 = 16 * 4
        self.latent_per_joint = latent_dim // num_joints  # 4

        # Joint embedding: 4 → joint_dim
        self.joint_embed = nn.Linear(self.latent_per_joint, joint_dim)

        # Positional encoding for 16 joints
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, joint_dim) * 0.02)

        # Self-Attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=joint_dim, nhead=num_heads,
                dim_feedforward=joint_dim*4, dropout=dropout, batch_first=True
            ) for _ in range(num_self_attn_layers)
        ])

        # HMD embedding: 9 → 3 tokens
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)
        )

        # Cross-Attention: joints query HMD
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(joint_dim)

        # Output projection
        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, z, hmd_info):
        B = z.size(0)

        # Z → joint tokens: [B, 64] → [B, 16, 4] → [B, 16, D]
        z_joints = z.view(B, -1, self.latent_per_joint)  # [B, 16, 4]
        joint_tokens = self.joint_embed(z_joints)  # [B, 16, D]
        joint_tokens = joint_tokens + self.pos_embed

        # Self-Attention (inter-joint relationships)
        for layer in self.self_attn_layers:
            joint_tokens = layer(joint_tokens)

        # HMD tokens: [B, 9] → [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross-Attention (joints ← HMD)
        cross_out, _ = self.cross_attn(
            query=joint_tokens, key=hmd_tokens, value=hmd_tokens
        )
        joint_tokens = self.cross_norm(joint_tokens + cross_out)

        # Output: [B, 16, 3]
        pose_3d = self.output_proj(joint_tokens)

        return pose_3d
```

---

### Loss Functions (Same as Baseline)

| Loss | Weight | Role |
|------|--------|------|
| `loss_kpt` (MSE) | 1000 | 2D heatmap supervision |
| `loss_heatmap_recon` (MSE) | 250 | Heatmap reconstruction |
| `loss_pose_l2norm` | 1.0 | 3D pose L2 distance |
| `loss_cosine_similarity` | 0.1 | Directional similarity |
| `loss_limb_length` | 0.25 | Limb length consistency |
| `loss_hmd` (MSE) | 1.0 | HMD reconstruction |

---

### Expected Effects

| Component | Baseline | Hybrid | Effect |
|-----------|----------|--------|--------|
| Heatmap→Z | Conv Encoder | **Same** | Stability maintained |
| Z utilization | Global (64) | Per-joint (16x4) | Per-joint information separation |
| Joint relations | Linear (implicit) | Self-Attn (explicit) | Skeleton structure learning |
| HMD fusion | Add | Cross-Attn | Selective reference |

**Expected**: Baseline-level stability + inter-joint relationship modeling → Target below 41mm

---

## Reference Papers

1. **Deep Mutual Learning** (Zhang et al., 2017) - arXiv:1706.00384
2. **Knowledge Distillation** (Hinton et al., 2015) - arXiv:1503.02531
3. **A Simple Baseline for 3D Pose** (Martinez et al., 2017) - ICCV 2017
4. **StyleGAN** (Karras et al., 2019) - CVPR 2019
5. **Lifting by Image** - arXiv:2312.15636

---

> **Experiment results**: See `EXPERIMENT_RESULTS.md`
> - Single COCO Baseline: **41.37mm** 🏆
> - Target: Achieve below 41mm
