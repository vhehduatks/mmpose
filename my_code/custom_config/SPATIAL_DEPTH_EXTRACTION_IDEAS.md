# Spatial Information Preserving Depth Extraction Methods

> Created: 2026-01-21

## Problem Definition

### Limitations of Current Phase 5-C Structure

```
Backbone [2048, 8, 8]
       |
       +-> Deconv -> Heatmap -> soft_argmax -> coords_2d (detach)
       |                                        |
       +-> GAP -> FC -> Z_backbone [256] -------+-> Lifting -> 3D
              ❌ Spatial information lost!       |
                                                +-> 3D Loss
```

**Problems**:
- Global Average Pooling compresses all spatial information into a mean
- To learn "the depth of the left elbow", the model needs to know **where the left elbow is**
- Z_backbone [256] only contains the global context of the entire image
- Result: MPJPE 176mm (more than 4x worse than baseline 41mm)

### Goal

```
Extract backbone features at each joint's 2D location
-> Enables per-joint depth cue learning
-> Preserves spatial information while maintaining 2D/3D role separation
```

---

## Method Comparison

| Method | Spatial Info | Implementation Complexity | Memory | Features |
|------|----------|------------|--------|------|
| GAP (current) | ❌ Lost | Low | Low | Failed |
| **1. Grid Sampling** | ✅ Precise | **Low** | Low | **Recommended** |
| 2. Heatmap Weighted | ✅ Soft | Low | Medium | Soft attention |
| 3. Cross-Attention | ✅ Learned | High | High | Flexible |
| 4. Deformable Attn | ✅ Learned | High | Medium | Sparse sampling |

---

## Method 1: Grid Sampling (Recommended)

### Core Idea

Use 2D coordinates to directly sample features at the corresponding locations from the backbone feature map.

```
coords_2d [B, 16, 2]  ->  backbone_feat [B, 2048, 8, 8]
         |                        |
         +------ grid_sample -----+
                      |
            joint_features [B, 16, 2048]
```

### Architecture Diagram

```
Backbone feat [B, 2048, 8, 8]
       |
       +-------------------------------------+
       |                                     |
       v (Deconv)                            |
Heatmap [B, 16, 47, 47]                      |
       |                                     |
       v (soft_argmax)                       |
coords_2d [B, 16, 2]                         |
       |                                     |
       | (detach)                            |
       |                                     |
       v                                     v
coords_2d_detached -----------------> grid_sample(backbone, coords)
                                             |
                                             v
                                  joint_features [B, 16, 2048]
                                             |
                                             v (FC)
                                  joint_depth [B, 16, 64]
                                             |
       +-------------------------------------+
       |
       v
Concat [coords_2d_detached, confidence, joint_depth, hmd_info]
       |
       v
Lifting Network -> 3D Pose
```

### Detailed Implementation

```python
import torch
import torch.nn.functional as F

class SpatialDepthExtractor(nn.Module):
    """
    Extract backbone features at 2D coordinate locations
    """
    def __init__(self, in_channels: int = 2048, out_channels: int = 64):
        super().__init__()
        # Compress per-joint features
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_feat: Tensor, coords_2d: Tensor) -> Tensor:
        """
        Args:
            backbone_feat: [B, C, H, W] backbone feature map
            coords_2d: [B, K, 2] normalized coordinates (0~1)

        Returns:
            joint_features: [B, K, out_channels] per-joint depth features
        """
        B, K, _ = coords_2d.shape

        # Step 1: Convert coords to grid_sample format
        # grid_sample uses [-1, 1] range
        coords_normalized = coords_2d * 2 - 1  # [0,1] -> [-1,1]

        # Step 2: Reshape to grid format
        # grid_sample expects: [B, H_out, W_out, 2]
        # We sample K points: [B, K, 1, 2]
        coords_grid = coords_normalized.unsqueeze(2)  # [B, K, 1, 2]

        # Step 3: Sample features via bilinear interpolation
        # backbone_feat: [B, C, H, W]
        # coords_grid: [B, K, 1, 2]
        # output: [B, C, K, 1]
        sampled = F.grid_sample(
            backbone_feat,
            coords_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Step 4: reshape to [B, K, C]
        sampled = sampled.squeeze(-1)  # [B, C, K]
        sampled = sampled.permute(0, 2, 1)  # [B, K, C]

        # Step 5: Reduce dimensions with FC
        joint_features = self.fc(sampled)  # [B, K, out_channels]

        return joint_features
```

### Gradient Flow Analysis

```
3D Loss
   |
   v
Lifting Network
   |
   +--- coords_2d_detached <-- coords_2d <-- heatmap
   |         ❌ gradient blocked          ✅ trained only by heatmap loss
   |
   +--- joint_depth
              |
              v (FC)
        joint_features
              |
              v (grid_sample)
        backbone_feat <-- Backbone
              ✅ 3D gradient flows!
```

**Key points**:
- coords_2d is detached -> heatmap is trained only by 2D loss
- grid_sample is differentiable -> 3D loss flows to the backbone
- Features at each joint's location are used to learn that joint's depth

### Coordinate System Conversion Details

```python
# soft_argmax output: [0, 1] range (normalized)
coords_2d = soft_argmax(heatmap)  # [B, K, 2], range [0, 1]

# grid_sample input: [-1, 1] range
#   (-1, -1) = top-left
#   (+1, +1) = bottom-right
coords_grid = coords_2d * 2 - 1  # [0,1] -> [-1,1]

# Example:
#   coords_2d = (0.0, 0.0) -> coords_grid = (-1, -1) -> top-left
#   coords_2d = (0.5, 0.5) -> coords_grid = (0, 0)   -> center
#   coords_2d = (1.0, 1.0) -> coords_grid = (1, 1)   -> bottom-right
```

### Backbone Feature Map Resolution

```
Backbone output: [B, 2048, 8, 8]
Heatmap: [B, 16, 47, 47]

Problem: Resolution mismatch
- coords_2d is based on 47x47 heatmap
- backbone is 8x8

Solution: grid_sample's bilinear interpolation
- Maps coords_2d [0,1] to 8x8
- Non-integer positions are handled by interpolation
- Example: (0.3, 0.7) -> (2.4, 5.6) on 8x8 -> interpolated from 4 neighboring pixels
```

### Modified Lifting Network

```python
class LiftingNetworkWithSpatialDepth(nn.Module):
    def __init__(self,
                 num_joints: int = 16,
                 hmd_dim: int = 9,
                 depth_dim: int = 64,  # joint_depth dimension
                 hidden_dim: int = 1024):
        super().__init__()

        # Input: coords(32) + conf(16) + joint_depth(16*64) + HMD(9)
        # = 32 + 16 + 1024 + 9 = 1081
        # Or process per-joint without flattening joint_depth

        # Option A: Flatten all
        input_dim = num_joints * 2 + num_joints + num_joints * depth_dim + hmd_dim

        # Option B: Per-joint processing (recommended)
        # Per joint: coords(2) + conf(1) + depth(64) = 67
        # -> Per-joint MLP -> concat -> final MLP
```

### Expected Advantages

1. **Spatial information preserved**: Each joint extracts features from its own location
2. **Per-joint depth learning**: 16 joints each have independent depth cues
3. **Gradient separation maintained**: coords_2d is still detached
4. **Simple implementation**: Solved with a single grid_sample

### Expected Issues and Solutions

| Issue | Solution |
|------|--------|
| 8x8 resolution too low | Use intermediate features (16x16, 32x32) |
| Single point sampling unstable | 3x3 region pooling or multi-scale |
| coords_2d error propagation | Temperature tuning for sharp heatmaps |

---

## Method 2: Heatmap-Weighted Pooling

### Idea

Use the heatmap itself as spatial attention to compute a weighted sum of backbone features.

```python
def heatmap_weighted_pooling(backbone_feat, heatmaps):
    """
    Args:
        backbone_feat: [B, C, H, W] - e.g., [B, 2048, 8, 8]
        heatmaps: [B, K, H', W'] - e.g., [B, 16, 47, 47]

    Returns:
        joint_features: [B, K, C]
    """
    B, C, H, W = backbone_feat.shape
    B, K, H2, W2 = heatmaps.shape

    # Resize heatmaps to backbone resolution
    heatmaps_resized = F.interpolate(
        heatmaps, size=(H, W), mode='bilinear', align_corners=True
    )  # [B, K, H, W]

    # Normalize to attention weights
    heatmaps_flat = heatmaps_resized.view(B, K, -1)  # [B, K, H*W]
    attn_weights = F.softmax(heatmaps_flat, dim=-1)   # [B, K, H*W]

    # Weighted sum
    backbone_flat = backbone_feat.view(B, C, -1)  # [B, C, H*W]

    # einsum: per-joint weighted sum
    joint_features = torch.einsum('bkn,bcn->bkc', attn_weights, backbone_flat)
    # [B, K, C]

    return joint_features
```

### Pros and Cons

**Pros**:
- Soft attention references a wider area
- Heatmap uncertainty is naturally reflected

**Cons**:
- If heatmap is detached, the attention is also fixed
- Higher computation cost than grid_sample

---

## Method 3: Cross-Attention (DETR Style)

### Idea

Transformer attention where each joint is used as a query, and backbone spatial features as key/value.

```python
class JointCrossAttention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.coord_embed = nn.Linear(2, d_model)
        self.backbone_proj = nn.Linear(2048, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, coords_2d, backbone_feat):
        """
        Args:
            coords_2d: [B, K, 2]
            backbone_feat: [B, C, H, W]
        """
        B, K, _ = coords_2d.shape

        # Query: joint position embeddings
        joint_queries = self.coord_embed(coords_2d)  # [B, K, D]

        # Key/Value: backbone spatial tokens
        backbone_tokens = backbone_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        backbone_kv = self.backbone_proj(backbone_tokens)  # [B, H*W, D]

        # Cross attention
        depth_features, attn_weights = self.cross_attn(
            query=joint_queries,
            key=backbone_kv,
            value=backbone_kv
        )  # [B, K, D], [B, K, H*W]

        return depth_features, attn_weights
```

### Pros and Cons

**Pros**:
- Learnable attention searches for optimal positions
- Can integrate information from multiple positions
- Interpretable through attention map visualization

**Cons**:
- Complex implementation
- Increased parameters
- Training can be unstable

---

## Method 4: Deformable Attention

### Idea

Like DETR's deformable attention, each joint sparsely samples multiple locations using learned offsets.

```python
# K sampling points per joint
# Offsets are learned to find optimal positions
sampling_offsets = self.offset_network(coords_2d)  # [B, K, num_points, 2]
sampling_locations = coords_2d.unsqueeze(2) + sampling_offsets

# Sample features at each location and compute weighted sum
```

### Pros and Cons

**Pros**: Efficient sparse sampling, flexible receptive field

**Cons**: Complex implementation, difficult training

---

## Implementation Priority

| Priority | Method | Reason |
|------|------|------|
| 1 | **Grid Sampling** | Simple, intuitive, quick validation |
| 2 | Heatmap Weighted | Soft version of Grid |
| 3 | Cross-Attention | When there is room for performance improvement |

---

## Next Steps

1. [ ] Implement Grid Sampling-based `CustomEgoposeSpatialLiftingHead`
2. [ ] Smoke test to verify training pipeline
3. [ ] Full dataset training and baseline comparison
4. [ ] Try Method 2, 3 if needed
