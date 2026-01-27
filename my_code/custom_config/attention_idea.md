


## Proposal: Attention-based Lifting Network

### Limitations of the Current Structure

```python
# Current: Simple concat → FC
x = concat([coords_2d, confidence, z_backbone, hmd])  # [B, 313]
pose_3d = FC(x)  # [B, 48]
```

**Problems**:
- Simple concat of all inputs → No relationship modeling
- Unclear which joints should reference which depth information
- Ignores structural relationships between joints (symmetry, parent-child)

### Option 1: Joint Self-Attention

```
2D coords [B, 16, 2] → Linear → Joint tokens [B, 16, D]
                                      ↓
                              Self-Attention
                              (Learn inter-joint relationships)
                                      ↓
                              [B, 16, D]
                                      ↓
                                  3D Pose
```

**Benefits**:
- Learn left arm ↔ right arm symmetry relationships
- Parent-child joint relationships (shoulder→elbow→wrist)
- Occluded joints can reference visible joints

---

### Option 2: Cross-Attention (2D → Backbone) ⭐ Recommended

**Core Idea**: Each joint's 2D position queries corresponding depth information from backbone features

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

**Implementation Example**:
```python
class CrossAttentionLifting(nn.Module):
    def __init__(self, joint_dim=64, backbone_dim=256, num_heads=4):
        super().__init__()

        # 2D coords → joint queries [B, 16, 64]
        self.joint_embed = nn.Sequential(
            nn.Linear(2, joint_dim),
            nn.ReLU()
        )

        # Backbone spatial tokens [B, 64, D]
        self.backbone_proj = nn.Conv2d(2048, backbone_dim, 1)
        self.kv_proj = nn.Linear(backbone_dim, joint_dim)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output
        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, coords_2d, backbone_feat, confidence, hmd_info):
        B = coords_2d.size(0)

        # Joint queries from 2D coords
        joint_q = self.joint_embed(coords_2d)  # [B, 16, 64]

        # Backbone spatial tokens
        backbone_tokens = self.backbone_proj(backbone_feat)  # [B, 256, 8, 8]
        backbone_tokens = backbone_tokens.flatten(2).transpose(1, 2)  # [B, 64, 256]
        backbone_kv = self.kv_proj(backbone_tokens)  # [B, 64, 64]

        # Cross attention: Each joint queries depth information from backbone
        depth_features, attn_weights = self.cross_attn(
            query=joint_q,      # [B, 16, 64] - Each joint
            key=backbone_kv,    # [B, 64, 64] - Spatial positions
            value=backbone_kv
        )
        # depth_features: [B, 16, 64] - depth-aware joint features
        # attn_weights: [B, 16, 64] - Can visualize which positions were referenced!

        # Combine with confidence and HMD
        # ... (additional processing)

        pose_3d = self.output_proj(depth_features)  # [B, 16, 3]
        return pose_3d, attn_weights
```

**Advantages**:
1. **Selective depth querying**: Each joint retrieves depth information from the spatial positions it needs
2. **Interpretable**: `attn_weights` can visualize "from which positions depth was retrieved"
3. **Matches role separation**: 2D=Query(position), Backbone=Key/Value(depth)

---

### Option 3: Joint-HMD Cross-Attention

```
Query: Joint tokens [B, 16, D]
Key/Value: HMD tokens [B, 3, D]  (head, right_hand, left_hand)
                ↓
        Cross-Attention
                ↓
    "Hand joints → reference hand HMD"
    "Torso joints → reference head HMD"
```

**Implementation**:
```python
class JointHMDCrossAttention(nn.Module):
    def __init__(self, joint_dim=64, num_heads=4):
        super().__init__()

        # HMD as 3 tokens (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, joint_dim * 3),  # → [B, 3, D]
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, joint_features, hmd_info):
        B = joint_features.size(0)

        # HMD → 3 tokens [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention: joints attend to HMD
        enhanced, _ = self.cross_attn(
            query=joint_features,  # [B, 16, D]
            key=hmd_tokens,        # [B, 3, D]
            value=hmd_tokens
        )

        return joint_features + enhanced  # Residual
```

**Effect**: Natural mapping of hand joints → hand HMD, torso → head HMD

---

### Full Attention Lifting Architecture (Recommended)

```
                    Backbone feat [2048, 8, 8]
                           │
         ┌─────────────────┴─────────────────┐
         ↓                                   ↓
    Heatmap [16,47,47]              Backbone tokens [64, D]
         ↓                                   │
    soft_argmax                              │
         ↓                                   │
   2D coords [16, 2]                         │
         ↓                                   │
   Joint embed [16, D]                       │
         │                                   │
         └───────── Cross-Attention ─────────┘
                   (Q: joints, K/V: backbone)
                           ↓
                  Depth-aware joints [16, D]
                           │
                           + HMD Cross-Attention
                           ↓
                    Self-Attention
                   (Inter-joint relationships)
                           ↓
                    Output proj
                           ↓
                    3D Pose [16, 3]
```

### Gradient Flow Design

```python
def forward(self, heatmaps, backbone_feat, hmd_info):
    # 2D coords (gradient blocked - 2D role only)
    coords_2d, confidence = soft_argmax_2d(heatmaps)
    coords_2d = coords_2d.detach()
    confidence = confidence.detach()

    # Joint queries
    joint_q = self.joint_embed(coords_2d)  # [B, 16, D]

    # Backbone key/value (gradient flows - 3D learning)
    backbone_kv = self.backbone_to_tokens(backbone_feat)  # [B, 64, D]

    # Cross attention: Depth query based on 2D positions
    depth_joints, attn = self.cross_attn(joint_q, backbone_kv, backbone_kv)

    # HMD fusion
    hmd_enhanced = self.hmd_cross_attn(depth_joints, hmd_info)

    # Self attention (inter-joint relationships)
    refined = self.self_attn(hmd_enhanced)

    # Output
    pose_3d = self.output_proj(refined)

    return pose_3d, attn  # attn enables interpretability
```

### Option Comparison

| Approach | Joint Relationships | Depth Query | Interpretable | Parameters |
|------|----------|-----------|----------|---------|
| FC (current) | ✗ | ✗ | ✗ | ~4M |
| Self-Attention | ✓ | ✗ | △ | ~5M |
| **Cross-Attention** | △ | **✓** | **✓** | ~6M |
| Full (Self+Cross+HMD) | ✓ | ✓ | ✓ | ~8M |

### Implementation Priority

1. **Cross-Attention (2D → Backbone)** - Core, implement first
2. HMD Cross-Attention - Additional improvement
3. Self-Attention - If needed

**Expected file**: `custom_egopose_attention_lifting_head.py`

---
