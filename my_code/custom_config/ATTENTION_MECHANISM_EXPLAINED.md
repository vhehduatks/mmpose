# Cross-Attention Mechanism Explanation

## 1. Basic Attention Concepts

### Self-Attention vs Cross-Attention

```
Self-Attention:
    Q, K, V are all generated from the same input
    Example: Learning relationships between words in a sentence

Cross-Attention:
    Q comes from one side, K/V come from the other side
    Example: In translation, referencing the source sentence (K/V) to generate target words (Q)
```

### Attention Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Q: Query  [B, N_q, D]  - The querying side
K: Key    [B, N_kv, D] - Search keys
V: Value  [B, N_kv, D] - Values to retrieve
```

---

## 2. Cross-Attention in Pose Estimation

### Purpose

```
"Each joint (Q) queries depth information from backbone features (K/V)"

Q: 2D joint positions → "What is the depth at this position?"
K/V: Backbone spatial features → "The depth information here is this"
```

### Structure

```
Backbone feat [2048, 8, 8]
       ↓ (Conv + flatten)
Backbone tokens [64, D]  ← K/V (8×8 = 64 spatial positions)

Heatmap → soft_argmax → 2D coords [16, 2]
       ↓ (Linear)
Joint tokens [16, D]     ← Q (16 joints)

Cross-Attention:
    Q: [16, D]   @ K^T: [D, 64]  = Scores: [16, 64]
    softmax(Scores) @ V: [64, D] = Output: [16, D]
```

---

## 3. Attention Computation Process (Detailed)

### Step 1: Preparing Query, Key, Value

```python
# Joint tokens (Query)
joint_tokens = joint_embed(coords_2d)  # [B, 16, D]
Q = joint_tokens  # [B, 16, 64]

# Backbone tokens (Key/Value)
backbone_feat = backbone(image)  # [B, 2048, 8, 8]
backbone_tokens = conv_proj(backbone_feat)  # [B, 128, 8, 8]
backbone_tokens = backbone_tokens.flatten(2).transpose(1, 2)  # [B, 64, 128]
K = V = linear_proj(backbone_tokens)  # [B, 64, 64]
```

### Step 2: Computing Attention Scores

```
Q @ K^T = [B, 16, 64] @ [B, 64, 64]^T = [B, 16, 64]

Score for how much each joint (16) should attend to each spatial position (64)

Example (Joint 0 = head):
    scores[0] = [0.5, 0.3, 0.1, 0.8, ..., 0.2]  # 64 scores
                 ↑                   ↑
              Position 0          Position 3
              (low relevance)     (high relevance)
```

### Step 3: Softmax Normalization

```
attention_weights = softmax(scores / √d, dim=-1)

Example (Joint 0 = head):
    weights[0] = [0.02, 0.01, 0.01, 0.15, ..., 0.03]  # sum = 1.0
                                  ↑
                          Highest weight
                          (References information from this position the most)
```

### Step 4: Weighted Sum of Values

```
output = attention_weights @ V
       = [B, 16, 64] @ [B, 64, 64]
       = [B, 16, 64]

Each joint computes a weighted average of values from 64 positions

Example (Joint 0 = head):
    output[0] = 0.02 * V[0] + 0.01 * V[1] + ... + 0.15 * V[3] + ...
                                                   ↑
                                        Most referenced position
```

---

## 4. Multi-Head Attention

### Concept

```
Single attention learns only one perspective
Multi-head learns from multiple perspectives simultaneously

Example: 4 heads
    Head 0: Learns vertical relationships
    Head 1: Learns horizontal relationships
    Head 2: Learns diagonal relationships
    Head 3: Learns overall context
```

### Implementation

```python
# D=64, num_heads=4 → head_dim = 64/4 = 16

Q, K, V: [B, N, 64]
    ↓ reshape
Q, K, V: [B, num_heads, N, head_dim] = [B, 4, N, 16]
    ↓ attention per head
outputs: [B, 4, N, 16]
    ↓ concat + linear
output: [B, N, 64]
```

---

## 5. Global Attention Lifting Structure

### Current Structure (1 Global Token)

```
K/V: [Backbone tokens + Global token]
     [B, 64, D]      +  [B, 1, D]   = [B, 65, D]
         ↑                  ↑
   8×8 spatial        Entire heatmap compressed
   positions          (HeatmapEncoder Z)
```

### Attention Visualization

```
                    K/V: 65 positions
     ┌─────────────────────────────────────────┐
     │  Backbone Spatial (64)      │ Global(1)│
     │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │    ┌──┐  │
     │  │  │  │  │  │  │  │  │  │  │    │ Z│  │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │    └──┘  │
     │  │  │  │  │  │  │  │  │  │  │          │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │          │
Q:   │  │  │  │  │  │  │  │  │  │  │          │
16   │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │          │
joints│  │  │  │  │  │  │  │  │  │  │          │
     │  └──┴──┴──┴──┴──┴──┴──┴──┘  │          │
     └─────────────────────────────────────────┘

Joint 0 (head)     → attend to 65 positions → output[0]
Joint 1 (neck)     → attend to 65 positions → output[1]
...
Joint 15 (ankle)   → attend to 65 positions → output[15]
```

### Problem

```
Among 65 K/V, Global Token is only 1 (1.5%)

Heatmap information of 16 joints compressed into 1 token
→ Information loss
→ Negligible influence of global token
```

---

## 6. Improved Structure: Per-Joint Heatmap Tokens

### New Structure

```
K/V: [Backbone tokens + Heatmap tokens]
     [B, 64, D]      +  [B, 16, D]   = [B, 80, D]
         ↑                  ↑
   8×8 spatial        Per-joint heatmap
   positions          (16 joints × each distribution)
```

### Per-Joint Heatmap Token Generation

```
Heatmap [B, 16, 47, 47]
    ↓ Per-joint processing
Joint 0 heatmap [B, 1, 47, 47] → Conv+Pool → Token 0 [B, 1, D]
Joint 1 heatmap [B, 1, 47, 47] → Conv+Pool → Token 1 [B, 1, D]
...
Joint 15 heatmap [B, 1, 47, 47] → Conv+Pool → Token 15 [B, 1, D]
    ↓ concat
Heatmap Tokens [B, 16, D]
```

### Improved Attention Visualization

```
                    K/V: 80 positions
     ┌──────────────────────────────────────────────────┐
     │  Backbone Spatial (64)      │ Heatmap (16)      │
     │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │ ┌──┬──┬──┬──┬──┐  │
     │  │  │  │  │  │  │  │  │  │  │ │H0│H1│H2│..│H15│ │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │ └──┴──┴──┴──┴──┘  │
     │  │  │  │  │  │  │  │  │  │  │       ↑           │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │  Per-joint token  │
Q:   │  │  │  │  │  │  │  │  │  │  │                   │
16   │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │                   │
joints│  │  │  │  │  │  │  │  │  │  │                   │
     │  └──┴──┴──┴──┴──┴──┴──┴──┘  │                   │
     └──────────────────────────────────────────────────┘

Joint 0 (head)  → attend to [spatial 64 + heatmap 16]
                  High attention expected on H0 (its own heatmap)

Joint 7 (wrist) → attend to [spatial 64 + heatmap 16]
                  High attention on H7 + referencing H6(elbow), H5(shoulder)
```

### Expected Benefits

```
1. Information preservation: Heatmap information of 16 joints is individually maintained
2. Self-reference: Joint i obtains its own distribution information from Heatmap Token i
3. Cross-reference: Occluded joints reference the heatmaps of visible joints
4. Greater proportion: 16 out of 80 = 20% (increased from 1.5%)
```

---

## 7. Code Example

### Per-Joint Heatmap Token Encoder

```python
class PerJointHeatmapEncoder(nn.Module):
    """Encodes each joint heatmap into an individual token"""

    def __init__(self, num_joints=16, output_dim=64):
        super().__init__()
        # Each joint heatmap [1, 47, 47] → [output_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [32, 23, 23]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 11, 11]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    def forward(self, heatmaps):
        """
        Args:
            heatmaps: [B, 16, 47, 47]
        Returns:
            tokens: [B, 16, D]
        """
        B, K, H, W = heatmaps.shape

        # Process each joint individually
        tokens = []
        for i in range(K):
            joint_hm = heatmaps[:, i:i+1, :, :]  # [B, 1, 47, 47]
            token = self.encoder(joint_hm)  # [B, D]
            tokens.append(token)

        tokens = torch.stack(tokens, dim=1)  # [B, 16, D]
        return tokens
```

### Cross-Attention with Heatmap Tokens

```python
class HeatmapBackboneCrossAttention(nn.Module):
    """Includes Backbone + Heatmap tokens in K/V"""

    def forward(self, joint_tokens, backbone_feat, heatmap_tokens):
        """
        Args:
            joint_tokens: [B, 16, D] - Query
            backbone_feat: [B, 2048, 8, 8]
            heatmap_tokens: [B, 16, D] - Per-joint heatmap
        """
        # Backbone → spatial tokens [B, 64, D]
        backbone_kv = self.backbone_proj(backbone_feat)

        # Concatenate: [B, 80, D]
        combined_kv = torch.cat([backbone_kv, heatmap_tokens], dim=1)

        # Cross attention
        output, attn_weights = self.cross_attn(
            query=joint_tokens,   # [B, 16, D]
            key=combined_kv,      # [B, 80, D]
            value=combined_kv
        )
        # attn_weights: [B, 16, 80]
        #   - [:, :, :64] → backbone spatial attention
        #   - [:, :, 64:] → heatmap token attention

        return output, attn_weights
```

---

## 8. Comparison Summary

| Structure | K/V Size | Heatmap Information | Expected Benefit |
|------|----------|-------------|----------|
| v1 (Baseline) | 64 | None | - |
| Global Token | 65 | 1 compressed (lossy) | Negligible |
| **Per-Joint Tokens** | **80** | **16 preserved** | **Expected improvement** |

---

## 9. Reference: Transformer Terminology

```
Encoder: Input → Representation (Self-Attention)
Decoder: Representation → Output (Cross-Attention referencing Encoder)

BERT: Encoder only (Bidirectional Self-Attention)
GPT:  Decoder only (Unidirectional Self-Attention)
T5:   Encoder-Decoder (Translation, etc.)

In Pose Estimation:
- Backbone = Encoder (Image → Feature)
- Lifting Network = Decoder (2D → 3D, references Backbone via Cross-Attention)
```
