# SPT, LSA, DWConv 상세 설명

> ViT Lifting v6에 적용된 Small Dataset 최적화 기법

## 배경: ViT의 문제점 (Small Dataset)

| 항목 | ViT 요구량 | EgoPose | 비율 |
|------|-----------|---------|------|
| 데이터 | 14M+ (ImageNet-21k) | 210K | **1.5%** |
| 권장 | 300M+ (JFT-300M) | 210K | **0.07%** |

ViT는 CNN의 **locality inductive bias** (이웃 pixel 우선 참조)가 없어서, 이를 데이터로부터 학습해야 합니다.
데이터가 부족하면 attention이 효과적으로 학습되지 않아 성능이 하락합니다.

**해결**: SPT + LSA + DWConv로 CNN의 locality bias를 ViT에 주입

---

## 1. SPT (Shifted Patch Tokenization)

### 문제: 1x1 Conv Tokenization은 이웃을 무시

```
기존 ViT Tokenization:

  Backbone feat [2048, 8, 8]
       ↓
  Conv2d(2048, 256, kernel=1)   ← 1x1 conv: 각 위치를 독립적으로 projection
       ↓
  Spatial Tokens [64, 256]      ← 각 토큰은 자기 위치 정보만 보유
```

1x1 Conv는 각 pixel을 **독립적으로** 처리합니다.
위치 `(3,4)`의 토큰은 `(3,4)`의 feature만 사용하고, 옆의 `(3,5)`, `(4,4)` 등 이웃 정보를 반영하지 않습니다.

CNN은 3x3 conv가 기본이라 항상 이웃 pixel을 참조하지만, ViT tokenization에는 이 **locality inductive bias가 없습니다**.

### 해결: 4방향 Shift 후 Concat

```
원본 Feature Map (8x8, 채널 C):

  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  |  | a| b| c|  |  |  |  |
  +--+--+--+--+--+--+--+--+     위치 (2,2)가 보는 정보:
  |  | d| *| e|  |  |  |  |
  +--+--+--+--+--+--+--+--+     기존 (1x1 Conv):    SPT (shift+concat):
  |  | f| g| h|  |  |  |  |
  +--+--+--+--+--+--+--+--+        +--+                +--+--+--+
  |  |  |  |  |  |  |  |  |        | *|  자기만         |  | b|  |
  +--+--+--+--+--+--+--+--+        +--+                +--+--+--+
                                                        | d| *| e|  자기+상하좌우
                                    Receptive: 1x1      +--+--+--+
                                                        |  | g|  |
                                                        +--+--+--+

                                                        Receptive: 3x3
```

### 구현 상세

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

        # 4방향 Shift
        x_left  = F.pad(x, (s, 0, 0, 0))[:, :, :, :W]   # 왼쪽으로 밀기
        x_right = F.pad(x, (0, s, 0, 0))[:, :, :, s:]    # 오른쪽으로 밀기
        x_up    = F.pad(x, (0, 0, s, 0))[:, :, :H, :]    # 위로 밀기
        x_down  = F.pad(x, (0, 0, 0, s))[:, :, s:, :]    # 아래로 밀기

        # 5개 방향 결합: [B, 2048*5, 8, 8] = [B, 10240, 8, 8]
        x_concat = torch.cat([x, x_left, x_right, x_up, x_down], dim=1)

        # Projection: [B, 10240, 8, 8] → [B, 128, 8, 8]
        return self.proj(x_concat)
```

### Shift 동작 원리

```
Left Shift 예시 (shift_size=1):

원본:                    Left Shift 결과:
+--+--+--+--+           +--+--+--+--+
| A| B| C| D|           | B| C| D| 0|    ← 한 칸 왼쪽으로 밀림
+--+--+--+--+           +--+--+--+--+
| E| F| G| H|           | F| G| H| 0|       오른쪽 경계는 0 (padding)
+--+--+--+--+           +--+--+--+--+

→ 위치 (0,0)에 원래 (0,1)=B의 값이 옴
→ 각 위치에 오른쪽 이웃의 정보가 담김

구현:
  F.pad(x, (1, 0, 0, 0))  → 왼쪽에 0 1열 추가: [A→0|A|B|C|D]
  [:, :, :, :W]            → 오른쪽 1열 잘라냄:  [0|A|B|C] → shift left 효과

  실제로는 (s, 0, 0, 0) = (left_pad, right_pad, top_pad, bottom_pad)
  왼쪽에 pad하고 오른쪽을 자르면 → 전체가 오른쪽으로 밀림
  → 각 위치에 왼쪽 이웃 정보가 담김
```

### Concat 후 각 위치의 정보

```
위치 (i, j)의 최종 토큰 = Concat([
    원본(i,j),       # 자기 자신         [C]
    Left(i,j),       # 왼쪽 이웃 (i,j-1)  [C]
    Right(i,j),      # 오른쪽 이웃 (i,j+1) [C]
    Up(i,j),         # 위 이웃 (i-1,j)    [C]
    Down(i,j)        # 아래 이웃 (i+1,j)  [C]
]) = [5C]

→ 1x1 Conv로 [5C] → [D] projection
→ 각 토큰이 이미 3x3 영역의 정보를 내포
```

### 효과

CNN의 3x3 conv와 유사한 **local receptive field**를 ViT tokenization에 부여.
Self-Attention이 이웃 관계를 처음부터 학습할 필요 없이, 토큰 자체에 이미 이웃 정보가 포함됨.

---

## 2. LSA (Locality Self-Attention)

### 문제: Standard Self-Attention의 균등 분산

```
Standard Self-Attention:

  Softmax temperature = sqrt(d_k)  ← 고정 상수

  Token 5의 attention weights 예시:
  [0.012, 0.013, 0.011, 0.015, 0.012, 0.72, 0.03, 0.04, ...]
                                        ^^^^
                                     자기 자신 (가장 높음!)

  문제 1: attention의 72%가 자기 자신에게 → 이웃 정보 수집 실패
  문제 2: 나머지 28%가 모든 토큰에 균등 분배 → 특정 이웃에 집중 못함
```

Small dataset에서는 attention이 어디를 봐야 하는지 학습하기 어렵습니다.

### 해결: 두 가지 핵심 기법

#### 기법 1: Learnable Temperature

```python
# Standard Self-Attention
attn = (Q @ K.T) / sqrt(d_k)               # sqrt(d_k) = 고정값 (예: 5.66)
attn = softmax(attn)

# LSA: Learnable Temperature
self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)

attn = (Q @ K.T) / sqrt(d_k)
attn = attn / self.temperature.clamp(min=0.1)  # temperature < 1 → 더 sharp
attn = softmax(attn)
```

Temperature가 softmax의 "날카로움"을 조절합니다:

```
Softmax 전 scores = [2.0, 1.5, 1.0, 0.5]:

  temp=1.0 (standard):   softmax → [0.33, 0.24, 0.18, 0.13]  (부드러운 분포)
  temp=0.5 (sharp):      softmax → [0.47, 0.27, 0.16, 0.09]  (날카로운 분포)
  temp=0.1 (very sharp): softmax → [0.85, 0.10, 0.03, 0.01]  (거의 argmax)
```

초기값 `0.5`로 처음부터 **sharp attention** → 가장 관련 있는 토큰에 집중.
학습하면서 temperature가 자동 조정됩니다.

```
학습 과정에서의 temperature 변화 (예상):

  초기 (epoch 0):   temperature ~ 0.5  → very sharp, 가까운 이웃 위주
  중반 (epoch 5):   temperature ~ 0.7  → 점점 부드러워짐
  후반 (epoch 10):  temperature ~ 1.0+ → 필요에 따라 global attention

  → 처음엔 local (CNN처럼), 나중엔 global (ViT 장점 활용)
     자동으로 최적 balance를 학습!
```

#### 기법 2: Diagonal Masking (자기 자신 제거)

```python
# Diagonal masking
diag_mask = torch.eye(N, device=x.device, dtype=torch.bool)  # [N, N] 단위 행렬
attn = attn.masked_fill(diag_mask, float('-inf'))
# softmax 후 대각선 = 0 (자기 자신 attention 제거)
```

왜 자기 자신을 제거하는가:

```
Standard Self-Attention에서:
  Q_i dot K_i 가 가장 높음 (자기 자신과의 내적이 최대)

  Token 5의 attention weights:
  Before: [0.05, 0.03, 0.04, 0.03, 0.04, [0.72], 0.03, ...]
                                           ^^^^^ Token 5 자기 자신

  문제: 72%가 자기 자신 → 이웃으로부터 새로운 정보를 거의 못 받음

  Diagonal Masking 적용:
  Mask:   [   ,    ,    ,    ,    , [-inf],    , ...]
  After:  [0.05, 0.03, 0.04, 0.03, 0.04, [-inf], 0.03, ...]

  Softmax 후:
          [0.12, 0.09, 0.14, 0.09, 0.12, [0.00], 0.11, ...]
                                           ^^^^^ 0으로 제거됨

  → 나머지 토큰들에게 attention이 재분배
  → 이웃으로부터 정보를 더 많이 흡수
```

### 전체 구현

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

        # 핵심 1: Learnable temperature
        attn = attn / self.temperature.clamp(min=0.1)

        # 핵심 2: Diagonal masking
        if use_diagonal_mask:
            mask = torch.eye(N, device=x.device, dtype=torch.bool)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)
```

### 효과

- **Learnable temperature**: 학습 초기에 sharp attention → local 패턴부터 학습
- **Diagonal masking**: 자기 참조 제거 → 이웃 정보 흡수 극대화
- 두 기법 합쳐서 Tiny-ImageNet +2.96%, CIFAR-100 +4% 성능 향상 (논문 결과)

---

## 3. DWConv (Depth-wise Convolution Embedding)

### 목적: Feature Map 단계에서 local 정보 추가 강화

SPT가 tokenization에서 locality를 추가하지만, **feature map 단계에서도 local feature를 강화**하면 더 효과적입니다.

### Standard Conv vs Depth-wise Conv

```
Standard Conv2d (kernel=3x3, in=128, out=128):

  128 input channels
       ↓
  128 x 128 x 3 x 3 filters   ← 파라미터: 128 x 128 x 9 = 147,456
       ↓
  128 output channels

  → 각 출력 채널이 모든 입력 채널을 조합 (cross-channel)

Depth-wise Conv2d (kernel=3x3, in=128, out=128, groups=128):

  128 input channels
   ↓    ↓    ↓    ↓
  ch0  ch1  ch2  ... ch127    ← 각 채널별 독립 3x3 conv
  3x3  3x3  3x3  ... 3x3     ← 파라미터: 128 x 9 = 1,152
   ↓    ↓    ↓    ↓
  128 output channels

  → 각 채널이 독립적으로 3x3 spatial filtering (cross-channel 없음)
```

| 항목 | Standard Conv | Depth-wise Conv | 비율 |
|------|--------------|-----------------|------|
| 파라미터 | 147,456 | **1,152** | **0.78%** |
| 연산량 | O(C^2 x K^2 x HW) | **O(C x K^2 x HW)** | **1/C** |
| 역할 | 채널 간 조합 + 공간 | **공간 정보만** | - |

### 구현

```python
class DepthWiseConvEmbedding(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1,
            groups=embed_dim,        # ← 핵심: groups=embed_dim → depth-wise
            bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, 128, 8, 8]
        return self.act(self.norm(self.dwconv(x))) + x   # Residual connection
        #      ^                                    ^
        #      local feature 강화                    원본 보존
```

### DWConv가 하는 일

```
입력 Feature Map의 한 채널 (8x8):

  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+
  |  | a| b| c|  |  |  |  |
  +--+--+--+--+--+--+--+--+     3x3 DWConv
  |  | d| *| e|  |  |  |  |  ──────────────→  *' = w1*a + w2*b + w3*c
  +--+--+--+--+--+--+--+--+                       + w4*d + w5*  + w6*e
  |  | f| g| h|  |  |  |  |                        + w7*f + w8*g + w9*h
  +--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |     ← 학습 가능한 3x3 spatial filter
  +--+--+--+--+--+--+--+--+        이웃 정보의 가중 합

  + Residual: *_out = *' + * (원본 보존)
```

각 채널별로 독립적인 3x3 spatial filtering → 이웃 pixel과의 관계를 local하게 학습

### 효과

- 파라미터 증가 최소화 (1,152개)하면서 local feature 강화
- Residual connection으로 원본 정보 보존
- SPT 이후 추가적인 local refinement 역할

---

## 전체 파이프라인에서의 역할

```
Backbone feat [2048, 8, 8]
       |
 +-------------------------------------+
 |  1. SPT                              |
 |  각 위치에 상하좌우 이웃을            |
 |  concat하여 tokenize                 |
 |  → "토큰이 이웃 정보를 내포"         |
 |  [2048, 8, 8] → [128, 8, 8]         |
 +-------------------------------------+
       |
 +-------------------------------------+
 |  2. DWConv                           |
 |  3x3 spatial filtering으로           |
 |  local feature 추가 강화             |
 |  → "더 정교한 이웃 관계 학습"        |
 |  [128, 8, 8] → [128, 8, 8]          |
 +-------------------------------------+
       |
 Flatten → Spatial Tokens [64, 128]
 + Joint Queries [16, 128]
       |
 +-------------------------------------+
 |  3. LSA                              |
 |  - Learnable temp: sharp attention   |
 |    → 가까운 토큰에 집중              |
 |  - Diagonal mask: 자기 제거          |
 |    → 이웃으로부터 정보 수집          |
 |  → "attention이 local부터 학습"      |
 +-------------------------------------+
       |
 Joint Tokens [16, 128]
       |
 3D Pose [16, 3]
```

## 각 기법의 역할 비유

| 기법 | 비유 | 역할 |
|------|------|------|
| **SPT** | 시험 전 양옆 친구 노트를 미리 복사 | **Input에 이웃 정보 사전 주입** |
| **DWConv** | 친구와 토론으로 이해력 향상 | **Feature를 local하게 정제** |
| **LSA** | 모르는 문제는 옆 사람 답 참고 (자기 답 X) | **Attention이 이웃부터 집중** |

세 기법 모두 **"ViT에 CNN의 locality inductive bias를 주입"**하는 목적이며,
데이터가 부족한 상황에서 Self-Attention이 local pattern부터 효과적으로 학습하도록 유도합니다.

---

## 참고 논문

- [Vision Transformer for Small-Size Datasets (AAAI 2022)](https://arxiv.org/abs/2112.13492) - SPT, LSA 제안
- [Depth-Wise Convolutions in ViTs (Neural Networks 2024)](https://www.sciencedirect.com/science/article/pii/S0925231224017697) - DWConv 적용
- [Graph-based ViT for Small Datasets (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-10408-0) - 최신 연구

## 적용 파일

| 파일 | 위치 |
|------|------|
| Head 구현 | `mmpose/models/heads/heatmap_heads/custom_egopose_vit_lifting_head_v6.py` |
| Smoke Config | `my_code/custom_config/HMD_xregopose_vit_lifting_v6_small_config.py` |
| Full Config | `my_code/custom_config/HMD_xregopose_vit_lifting_v6_full_config.py` |
