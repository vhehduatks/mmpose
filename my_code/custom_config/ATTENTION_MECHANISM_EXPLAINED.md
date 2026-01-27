# Cross-Attention 메커니즘 설명

## 1. 기본 Attention 개념

### Self-Attention vs Cross-Attention

```
Self-Attention:
    Q, K, V가 모두 같은 입력에서 생성
    예: 문장 내 단어들 간의 관계 학습

Cross-Attention:
    Q는 한 쪽, K/V는 다른 쪽에서 생성
    예: 번역 시 source 문장(K/V)을 참조하여 target 단어(Q) 생성
```

### Attention 수식

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Q: Query  [B, N_q, D]  - 질문하는 쪽
K: Key    [B, N_kv, D] - 검색 키
V: Value  [B, N_kv, D] - 가져올 값
```

---

## 2. Pose Estimation에서의 Cross-Attention

### 목적

```
"각 관절(Q)이 backbone feature(K/V)에서 depth 정보를 쿼리"

Q: 2D 관절 위치 → "이 위치의 depth는?"
K/V: Backbone spatial features → "여기의 depth 정보는 이거야"
```

### 구조

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

## 3. Attention 계산 과정 (상세)

### Step 1: Query, Key, Value 준비

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

### Step 2: Attention Score 계산

```
Q @ K^T = [B, 16, 64] @ [B, 64, 64]^T = [B, 16, 64]

각 관절(16)이 각 spatial 위치(64)에 얼마나 attend할지 점수

예시 (Joint 0 = head):
    scores[0] = [0.5, 0.3, 0.1, 0.8, ..., 0.2]  # 64개 점수
                 ↑                   ↑
              위치 0              위치 3
              (낮은 관련성)        (높은 관련성)
```

### Step 3: Softmax 정규화

```
attention_weights = softmax(scores / √d, dim=-1)

예시 (Joint 0 = head):
    weights[0] = [0.02, 0.01, 0.01, 0.15, ..., 0.03]  # 합 = 1.0
                                  ↑
                            가장 높은 가중치
                            (이 위치의 정보를 가장 많이 참조)
```

### Step 4: Value 가중합

```
output = attention_weights @ V
       = [B, 16, 64] @ [B, 64, 64]
       = [B, 16, 64]

각 관절이 64개 위치의 value를 가중 평균

예시 (Joint 0 = head):
    output[0] = 0.02 * V[0] + 0.01 * V[1] + ... + 0.15 * V[3] + ...
                                                   ↑
                                        가장 많이 참조한 위치
```

---

## 4. Multi-Head Attention

### 개념

```
단일 attention은 하나의 관점만 학습
Multi-head는 여러 관점에서 동시에 학습

예: 4 heads
    Head 0: 수직 방향 관계 학습
    Head 1: 수평 방향 관계 학습
    Head 2: 대각선 관계 학습
    Head 3: 전체 context 학습
```

### 구현

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

## 5. Global Attention Lifting 구조

### 현재 구조 (Global Token 1개)

```
K/V: [Backbone tokens + Global token]
     [B, 64, D]      +  [B, 1, D]   = [B, 65, D]
         ↑                  ↑
   8×8 spatial        Heatmap 전체 압축
   positions          (HeatmapEncoder Z)
```

### Attention 시각화

```
                    K/V: 65개 위치
     ┌─────────────────────────────────────────┐
     │  Backbone Spatial (64)      │ Global(1)│
     │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │    ┌──┐  │
     │  │  │  │  │  │  │  │  │  │  │    │ Z│  │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │    └──┘  │
     │  │  │  │  │  │  │  │  │  │  │          │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │          │
Q:   │  │  │  │  │  │  │  │  │  │  │          │
16개 │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │          │
관절 │  │  │  │  │  │  │  │  │  │  │          │
     │  └──┴──┴──┴──┴──┴──┴──┴──┘  │          │
     └─────────────────────────────────────────┘

Joint 0 (head)     → attend to 65 positions → output[0]
Joint 1 (neck)     → attend to 65 positions → output[1]
...
Joint 15 (ankle)   → attend to 65 positions → output[15]
```

### 문제점

```
65개 K/V 중 Global Token은 단 1개 (1.5%)

16개 관절의 heatmap 정보가 1개 token에 압축
→ 정보 손실
→ Global token의 영향력 미미
```

---

## 6. 개선된 구조: Per-Joint Heatmap Tokens

### 새 구조

```
K/V: [Backbone tokens + Heatmap tokens]
     [B, 64, D]      +  [B, 16, D]   = [B, 80, D]
         ↑                  ↑
   8×8 spatial        각 관절별 heatmap
   positions          (16개 관절 × 각각의 분포)
```

### Per-Joint Heatmap Token 생성

```
Heatmap [B, 16, 47, 47]
    ↓ 각 관절별 처리
Joint 0 heatmap [B, 1, 47, 47] → Conv+Pool → Token 0 [B, 1, D]
Joint 1 heatmap [B, 1, 47, 47] → Conv+Pool → Token 1 [B, 1, D]
...
Joint 15 heatmap [B, 1, 47, 47] → Conv+Pool → Token 15 [B, 1, D]
    ↓ concat
Heatmap Tokens [B, 16, D]
```

### 개선된 Attention 시각화

```
                    K/V: 80개 위치
     ┌──────────────────────────────────────────────────┐
     │  Backbone Spatial (64)      │ Heatmap (16)      │
     │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │ ┌──┬──┬──┬──┬──┐  │
     │  │  │  │  │  │  │  │  │  │  │ │H0│H1│H2│..│H15│ │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │ └──┴──┴──┴──┴──┘  │
     │  │  │  │  │  │  │  │  │  │  │       ↑           │
     │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │  각 관절별 token  │
Q:   │  │  │  │  │  │  │  │  │  │  │                   │
16개 │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │                   │
관절 │  │  │  │  │  │  │  │  │  │  │                   │
     │  └──┴──┴──┴──┴──┴──┴──┴──┘  │                   │
     └──────────────────────────────────────────────────┘

Joint 0 (head)  → attend to [spatial 64 + heatmap 16]
                  특히 H0 (자신의 heatmap)에 높은 attention 기대

Joint 7 (wrist) → attend to [spatial 64 + heatmap 16]
                  H7에 높은 attention + H6(elbow), H5(shoulder) 참조
```

### 기대 효과

```
1. 정보 보존: 16개 관절 heatmap 정보가 각각 유지
2. 자기 참조: Joint i가 Heatmap Token i에서 자신의 분포 정보 획득
3. 상호 참조: 가려진 관절이 보이는 관절의 heatmap 참조
4. 더 큰 비중: 80개 중 16개 = 20% (기존 1.5%에서 증가)
```

---

## 7. 코드 예시

### Per-Joint Heatmap Token Encoder

```python
class PerJointHeatmapEncoder(nn.Module):
    """각 관절 heatmap을 개별 token으로 인코딩"""

    def __init__(self, num_joints=16, output_dim=64):
        super().__init__()
        # 각 관절 heatmap [1, 47, 47] → [output_dim]
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

        # 각 관절별로 처리
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
    """K/V에 Backbone + Heatmap tokens 포함"""

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

## 8. 비교 요약

| 구조 | K/V 크기 | Heatmap 정보 | 기대 효과 |
|------|----------|-------------|----------|
| v1 (Baseline) | 64 | 없음 | - |
| Global Token | 65 | 1개 압축 (손실) | 미미 |
| **Per-Joint Tokens** | **80** | **16개 보존** | **향상 기대** |

---

## 9. 참고: Transformer 용어

```
Encoder: 입력 → 표현 (Self-Attention)
Decoder: 표현 → 출력 (Cross-Attention으로 Encoder 참조)

BERT: Encoder only (양방향 Self-Attention)
GPT:  Decoder only (단방향 Self-Attention)
T5:   Encoder-Decoder (번역 등)

Pose Estimation에서:
- Backbone = Encoder (이미지 → feature)
- Lifting Network = Decoder (2D → 3D, Cross-Attention으로 Backbone 참조)
```
