# SPT, LSA, DWConv Detailed Explanation

> Small Dataset optimization techniques applied to ViT Lifting v6

## Background: ViT's Problems (Small Dataset)

| Item | ViT Requirement | EgoPose | Ratio |
|------|-----------|---------|------|
| Data | 14M+ (ImageNet-21k) | 210K | **1.5%** |
| Recommended | 300M+ (JFT-300M) | 210K | **0.07%** |

ViT lacks CNN's **locality inductive bias** (prioritizing neighboring pixels), so it must learn this from data.
When data is insufficient, attention cannot be learned effectively, leading to performance degradation.

**Solution**: Inject CNN's locality bias into ViT via SPT + LSA + DWConv

---

## 1. SPT (Shifted Patch Tokenization)

### Problem: 1x1 Conv Tokenization Ignores Neighbors

```
Existing ViT Tokenization:

  Backbone feat [2048, 8, 8]
       ↓
  Conv2d(2048, 256, kernel=1)   ← 1x1 conv: projects each position independently
       ↓
  Spatial Tokens [64, 256]      ← Each token only holds information from its own position
```

1x1 Conv processes each pixel **independently**.
The token at position `(3,4)` only uses the feature at `(3,4)`, and does not incorporate neighbor information from `(3,5)`, `(4,4)`, etc.

CNNs use 3x3 conv as the default so they always reference neighboring pixels, but ViT tokenization lacks this **locality inductive bias**.

### Solution: 4-directional Shift then Concat

```
Original Feature Map (8x8, channel C):

  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  |  | a| b| c|  |  |  |  |
  +--+--+--+--+--+--+--+--+     Information seen by position (2,2):
  |  | d| *| e|  |  |  |  |
  +--+--+--+--+--+--+--+--+     Original (1x1 Conv):    SPT (shift+concat):
  |  | f| g| h|  |  |  |  |
  +--+--+--+--+--+--+--+--+        +--+                +--+--+--+
  |  |  |  |  |  |  |  |  |        | *|  Self only      |  | b|  |
  +--+--+--+--+--+--+--+--+        +--+                +--+--+--+
                                                        | d| *| e|  Self+up/down/left/right
                                    Receptive: 1x1      +--+--+--+
                                                        |  | g|  |
                                                        +--+--+--+

                                                        Receptive: 3x3
```

### Implementation Details

```python
class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_channels, embed_dim, shift_size=1):
        super().__init__()
        self.shift_size = shift_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 5, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape   # [B, 2048, 8, 8]
        s = self.shift_size     # 1 pixel shift

        # 4-directional Shift
        x_left  = F.pad(x, (s, 0, 0, 0))[:, :, :, :W]   # Shift left
        x_right = F.pad(x, (0, s, 0, 0))[:, :, :, s:]    # Shift right
        x_up    = F.pad(x, (0, 0, s, 0))[:, :, :H, :]    # Shift up
        x_down  = F.pad(x, (0, 0, 0, s))[:, :, s:, :]    # Shift down

        # Combine 5 directions: [B, 2048*5, 8, 8] = [B, 10240, 8, 8]
        x_concat = torch.cat([x, x_left, x_right, x_up, x_down], dim=1)

        # Projection: [B, 10240, 8, 8] → [B, 128, 8, 8]
        return self.proj(x_concat)
```

### How Shift Works

```
Left Shift Example (shift_size=1):

Original:                    Left Shift Result:
+--+--+--+--+           +--+--+--+--+
| A| B| C| D|           | B| C| D| 0|    ← Shifted one position to the left
+--+--+--+--+           +--+--+--+--+
| E| F| G| H|           | F| G| H| 0|       Right boundary is 0 (padding)
+--+--+--+--+           +--+--+--+--+

→ Position (0,0) now has the value from original (0,1)=B
→ Each position contains its right neighbor's information

Implementation:
  F.pad(x, (1, 0, 0, 0))  → Add 1 column of 0s on the left: [A→0|A|B|C|D]
  [:, :, :, :W]            → Trim 1 column from the right:  [0|A|B|C] → shift left effect

  Actually (s, 0, 0, 0) = (left_pad, right_pad, top_pad, bottom_pad)
  Padding on the left and trimming the right → entire content shifts to the right
  → Each position contains left neighbor's information
```

### Information at Each Position After Concat

```
Final token at position (i, j) = Concat([
    Original(i,j),   # Self                   [C]
    Left(i,j),       # Left neighbor (i,j-1)  [C]
    Right(i,j),      # Right neighbor (i,j+1) [C]
    Up(i,j),         # Upper neighbor (i-1,j) [C]
    Down(i,j)        # Lower neighbor (i+1,j) [C]
]) = [5C]

→ 1x1 Conv projects [5C] → [D]
→ Each token already contains information from a 3x3 region
```

### Effect

Provides a **local receptive field** similar to CNN's 3x3 conv to ViT tokenization.
Self-Attention does not need to learn neighbor relationships from scratch, as neighbor information is already embedded in the tokens.

---

## 2. LSA (Locality Self-Attention)

### Problem: Uniform Distribution of Standard Self-Attention

```
Standard Self-Attention:

  Softmax temperature = sqrt(d_k)  ← Fixed constant

  Example attention weights for Token 5:
  [0.012, 0.013, 0.011, 0.015, 0.012, 0.72, 0.03, 0.04, ...]
                                        ^^^^
                                     Self (highest!)

  Problem 1: 72% of attention goes to itself → Fails to collect neighbor information
  Problem 2: Remaining 28% distributed uniformly across all tokens → Cannot focus on specific neighbors
```

On small datasets, attention has difficulty learning where to look.

### Solution: Two Key Techniques

#### Technique 1: Learnable Temperature

```python
# Standard Self-Attention
attn = (Q @ K.T) / sqrt(d_k)               # sqrt(d_k) = fixed value (e.g., 5.66)
attn = softmax(attn)

# LSA: Learnable Temperature
self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)

attn = (Q @ K.T) / sqrt(d_k)
attn = attn / self.temperature.clamp(min=0.1)  # temperature < 1 → sharper
attn = softmax(attn)
```

Temperature controls the "sharpness" of softmax:

```
Pre-softmax scores = [2.0, 1.5, 1.0, 0.5]:

  temp=1.0 (standard):   softmax → [0.33, 0.24, 0.18, 0.13]  (smooth distribution)
  temp=0.5 (sharp):      softmax → [0.47, 0.27, 0.16, 0.09]  (sharp distribution)
  temp=0.1 (very sharp): softmax → [0.85, 0.10, 0.03, 0.01]  (nearly argmax)
```

Initial value `0.5` provides **sharp attention** from the start → focuses on the most relevant tokens.
Temperature is automatically adjusted during training.

```
Expected temperature changes during training:

  Early (epoch 0):   temperature ~ 0.5  → very sharp, focuses on nearby neighbors
  Mid (epoch 5):     temperature ~ 0.7  → gradually becomes smoother
  Late (epoch 10):   temperature ~ 1.0+ → global attention as needed

  → Local first (like CNN), then global (leveraging ViT's strengths)
     Automatically learns the optimal balance!
```

#### Technique 2: Diagonal Masking (Removing Self-Attention)

```python
# Diagonal masking
diag_mask = torch.eye(N, device=x.device, dtype=torch.bool)  # [N, N] identity matrix
attn = attn.masked_fill(diag_mask, float('-inf'))
# After softmax, diagonal = 0 (self-attention removed)
```

Why remove self-attention:

```
In Standard Self-Attention:
  Q_i dot K_i is the highest (dot product with itself is maximum)

  Token 5's attention weights:
  Before: [0.05, 0.03, 0.04, 0.03, 0.04, [0.72], 0.03, ...]
                                           ^^^^^ Token 5 itself

  Problem: 72% goes to itself → Barely receives new information from neighbors

  With Diagonal Masking:
  Mask:   [   ,    ,    ,    ,    , [-inf],    , ...]
  After:  [0.05, 0.03, 0.04, 0.03, 0.04, [-inf], 0.03, ...]

  After Softmax:
          [0.12, 0.09, 0.14, 0.09, 0.12, [0.00], 0.11, ...]
                                           ^^^^^ Removed to 0

  → Attention is redistributed to other tokens
  → Absorbs more information from neighbors
```

### Full Implementation

```python
class LocalitySelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, init_temperature=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Per-head learnable temperature
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1) * init_temperature
        )

    def forward(self, x, use_diagonal_mask=True):
        B, N, D = x.shape

        # QKV: [B, N, 3D] → [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Key point 1: Learnable temperature
        attn = attn / self.temperature.clamp(min=0.1)

        # Key point 2: Diagonal masking
        if use_diagonal_mask:
            mask = torch.eye(N, device=x.device, dtype=torch.bool)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)
```

### Effect

- **Learnable temperature**: Sharp attention early in training → learns local patterns first
- **Diagonal masking**: Removes self-reference → maximizes information absorption from neighbors
- Combined, the two techniques achieve +2.96% on Tiny-ImageNet, +4% on CIFAR-100 (paper results)

---

## 3. DWConv (Depth-wise Convolution Embedding)

### Purpose: Further Strengthening Local Information at Feature Map Stage

While SPT adds locality at the tokenization stage, it is even more effective to **strengthen local features at the feature map stage** as well.

### Standard Conv vs Depth-wise Conv

```
Standard Conv2d (kernel=3x3, in=128, out=128):

  128 input channels
       ↓
  128 x 128 x 3 x 3 filters   ← Parameters: 128 x 128 x 9 = 147,456
       ↓
  128 output channels

  → Each output channel combines all input channels (cross-channel)

Depth-wise Conv2d (kernel=3x3, in=128, out=128, groups=128):

  128 input channels
   ↓    ↓    ↓    ↓
  ch0  ch1  ch2  ... ch127    ← Independent 3x3 conv per channel
  3x3  3x3  3x3  ... 3x3     ← Parameters: 128 x 9 = 1,152
   ↓    ↓    ↓    ↓
  128 output channels

  → Each channel independently performs 3x3 spatial filtering (no cross-channel)
```

| Item | Standard Conv | Depth-wise Conv | Ratio |
|------|--------------|-----------------|------|
| Parameters | 147,456 | **1,152** | **0.78%** |
| Computation | O(C^2 x K^2 x HW) | **O(C x K^2 x HW)** | **1/C** |
| Role | Cross-channel + spatial | **Spatial only** | - |

### Implementation

```python
class DepthWiseConvEmbedding(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1,
            groups=embed_dim,        # ← Key: groups=embed_dim → depth-wise
            bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, 128, 8, 8]
        return self.act(self.norm(self.dwconv(x))) + x   # Residual connection
        #      ^                                    ^
        #      Local feature enhancement            Original preservation
```

### What DWConv Does

```
One channel of the input Feature Map (8x8):

  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  |  | a| b| c|  |  |  |  |
  +--+--+--+--+--+--+--+--+     3x3 DWConv
  |  | d| *| e|  |  |  |  |  ──────────────→  *' = w1*a + w2*b + w3*c
  +--+--+--+--+--+--+--+--+                       + w4*d + w5*  + w6*e
  |  | f| g| h|  |  |  |  |                        + w7*f + w8*g + w9*h
  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |     ← Learnable 3x3 spatial filter
  +--+--+--+--+--+--+--+--+        Weighted sum of neighbor information

  + Residual: *_out = *' + * (original preserved)
```

Independent 3x3 spatial filtering per channel → locally learns relationships with neighboring pixels

### Effect

- Minimizes parameter increase (1,152 parameters) while enhancing local features
- Residual connection preserves original information
- Serves as additional local refinement after SPT

---

## Role in the Overall Pipeline

```
Backbone feat [2048, 8, 8]
       |
 +-------------------------------------+
 |  1. SPT                              |
 |  Concatenates up/down/left/right     |
 |  neighbors at each position to       |
 |  tokenize                            |
 |  → "Tokens embed neighbor info"      |
 |  [2048, 8, 8] → [128, 8, 8]         |
 +-------------------------------------+
       |
 +-------------------------------------+
 |  2. DWConv                           |
 |  3x3 spatial filtering to            |
 |  further strengthen local features   |
 |  → "More refined neighbor learning"  |
 |  [128, 8, 8] → [128, 8, 8]          |
 +-------------------------------------+
       |
 Flatten → Spatial Tokens [64, 128]
 + Joint Queries [16, 128]
       |
 +-------------------------------------+
 |  3. LSA                              |
 |  - Learnable temp: sharp attention   |
 |    → Focuses on nearby tokens        |
 |  - Diagonal mask: removes self       |
 |    → Collects info from neighbors    |
 |  → "Attention learns local first"    |
 +-------------------------------------+
       |
 Joint Tokens [16, 128]
       |
 3D Pose [16, 3]
```

## Analogy for Each Technique's Role

| Technique | Analogy | Role |
|------|------|------|
| **SPT** | Copying neighbors' notes before an exam | **Pre-injecting neighbor information into input** |
| **DWConv** | Improving understanding through discussion with friends | **Locally refining features** |
| **LSA** | Referencing neighbors' answers for unknown questions (not your own) | **Attention focuses on neighbors first** |

All three techniques share the purpose of **"injecting CNN's locality inductive bias into ViT"**,
guiding Self-Attention to effectively learn local patterns first when data is insufficient.

---

## Reference Papers

- [Vision Transformer for Small-Size Datasets (AAAI 2022)](https://arxiv.org/abs/2112.13492) - Proposes SPT, LSA
- [Depth-Wise Convolutions in ViTs (Neural Networks 2024)](https://www.sciencedirect.com/science/article/pii/S0925231224017697) - DWConv application
- [Graph-based ViT for Small Datasets (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-10408-0) - Latest research

## Applied Files

| File | Location |
|------|------|
| Head implementation | `mmpose/models/heads/heatmap_heads/custom_egopose_vit_lifting_head_v6.py` |
| Smoke Config | `my_code/custom_config/HMD_xregopose_vit_lifting_v6_small_config.py` |
| Full Config | `my_code/custom_config/HMD_xregopose_vit_lifting_v6_full_config.py` |
