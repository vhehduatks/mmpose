# ViT-Style Attention Lifting for 3D Pose Estimation (v2)

## 0. Key Insight from EfficientHeatmapDecoder

```
EfficientHeatmapDecoder의 핵심:
  Heatmap reconstruction의 목적은 heatmap을 디코딩하는 것이 아님!
  → Latent vector Z에 관절 정보를 주입하는 것이 목적

  Reconstruction loss가 Z를 regularize → Z가 관절 위치 정보를 포함하도록 강제
```

**이 원리를 ViT-Style에 적용:**
- Joint tokens = Latent representation
- Heatmap Reconstruction = 2D 관절 정보를 joint tokens에 주입
- HMD Cross-Attention = 3D depth 정보 추가

---

## 1. 배경

### 기존 Attention Lifting의 문제점

```
현재 구조:
  Heatmap → soft_argmax → 2D coords [B, 16, 2]
      ↓
  joint_embed(coords) → Q [B, 16, D]
      ↓
  Cross-Attention(Q, K=backbone, V=backbone)
      ↓
  Depth prediction
```

**문제점:**
1. `soft_argmax`는 (x, y) 좌표만 추출 → spatial feature 정보 손실
2. Query가 좌표 기반이라 표현력 제한
3. 단방향 정보 흐름 (backbone → joint)

---

## 2. ViTPose 참고 (arxiv 2204.12484)

### ViTPose 핵심 구조

```
Image → Patch Embedding → [B, 256, D] (16×16 patches)
    ↓
Transformer Encoder (Self-Attention)
    ↓
Simple Decoder → Heatmaps
```

### Patch Token의 장점

1. **위치 정보**: Positional encoding으로 spatial 위치 보존
2. **Local + Global**: Self-attention으로 전체 context 획득
3. **Semantic 정보**: 학습된 feature representation

---

## 3. 제안: ViT-Style Lifting v2 (Reconstruction Regularized)

### 핵심 아이디어

```
soft_argmax로 좌표 추출 대신 → Learnable Joint Queries 사용
Cross-Attention 대신 → Self-Attention (joint ↔ patch 양방향)
Heatmap Reconstruction → 2D 관절 정보를 joint tokens에 주입
HMD Cross-Attention → 3D depth reference 추가
```

### 구조 (v2)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Backbone (ResNet) → Feature [B, 2048, 8, 8]               │
│         ↓                                                   │
│  Projection → Spatial Tokens [B, 64, D]                    │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │ Concat: [Spatial Tokens] + [Joint Queries]              │
│  │         [B, 64, D]      + [B, 16, D]    │                │
│  │              ↓                          │                │
│  │         [B, 80, D]                      │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │     Self-Attention (4 layers)           │                │
│  │     - Joint tokens ↔ Spatial tokens     │                │
│  │     - 양방향 정보 교환                    │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Joint Tokens [B, 16, D] (latent representation)           │
│         │                                                   │
│         ├───────────────────────┐                           │
│         │                       ↓                           │
│         │              ┌────────────────────┐               │
│         │              │  Heatmap Decoder   │               │
│         │              │  (Per-Joint)       │               │
│         │              └────────────────────┘               │
│         │                       ↓                           │
│         │              Recon Heatmap [16, 47, 47]           │
│         │              (loss: 2D 정보를 tokens에 주입)       │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │     HMD Cross-Attention                 │                │
│  │     (3D depth reference 추가)            │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  ┌──────────────────────────────┐                           │
│  │     3D Pose Head             │                           │
│  │     Linear(D, 3)             │                           │
│  └──────────────────────────────┘                           │
│         ↓                                                   │
│    3D Pose [B, 16, 3]                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 정보 흐름

```
1. Self-Attention: Spatial ↔ Joint 양방향 상호작용
2. Heatmap Reconstruction: 2D 관절 위치 정보를 tokens에 강제 주입
3. HMD Cross-Attention: 3D depth reference (머리/손 위치) 추가
4. 3D Pose Head: 풍부한 2D+3D 정보로 최종 3D 포즈 예측
```

---

## 4. 기존 대비 변경점

| 항목 | 기존 Attention Lifting | ViT-Style Lifting v2 |
|------|------------------------|----------------------|
| Query 생성 | soft_argmax(heatmap) → embed | **Learnable parameters** |
| Attention | Cross-Attention | **Self-Attention + HMD Cross-Attn** |
| 정보 흐름 | Backbone → Joint (단방향) | **Backbone ↔ Joint (양방향)** |
| Heatmap 역할 | soft_argmax 입력 | **Reconstruction regularization** |
| 3D 예측 | 2D + Depth concat | **Direct 3D Head** |
| 2D 정보 주입 | soft_argmax 좌표 | **Reconstruction loss** |

---

## 5. Learnable Joint Queries

### 개념

```python
# 각 관절마다 학습 가능한 query vector
self.joint_queries = nn.Parameter(torch.randn(1, 16, D))

# soft_argmax 대신:
Q = self.joint_queries.expand(B, -1, -1)  # [B, 16, D]
```

### 장점

1. **End-to-end 학습**: 좌표 추출 병목 없음
2. **풍부한 표현**: D차원 전체 활용 (vs 2D 좌표)
3. **검증된 방식**: DETR, TokenPose에서 효과 입증

### 학습 과정

```
초기: random initialization
학습 후: 각 joint query가 해당 관절을 찾는 "질문" 학습

joint_queries[0] → "머리는 어디?" 라는 질문 encoding
joint_queries[7] → "오른쪽 손목은 어디?" 라는 질문 encoding
```

---

## 6. Self-Attention의 역할

### Cross-Attention vs Self-Attention

```
Cross-Attention (기존):
  Q: Joint tokens
  K, V: Spatial tokens
  → Joint가 Spatial을 참조 (단방향)

Self-Attention (제안):
  Q, K, V: [Spatial; Joint] 전체
  → Spatial ↔ Joint 양방향 + Spatial ↔ Spatial + Joint ↔ Joint
```

### 양방향의 이점

```
Joint → Spatial: "이 위치에 관절이 있나?" (depth 정보 획득)
Spatial → Joint: "이 관절 정보를 반영해서 feature 업데이트"
Joint → Joint: "다른 관절과의 관계 학습" (skeleton prior)
```

---

## 7. 기대 효과

1. **정보 손실 감소**: soft_argmax 좌표 추출 병목 제거
2. **풍부한 Query**: 학습된 embedding이 좌표보다 표현력 높음
3. **Skeleton 관계 학습**: Joint ↔ Joint self-attention
4. **2D 정보 강제 주입**: Reconstruction loss로 tokens에 관절 위치 정보 보장
5. **명확한 정보 분리**: 2D (reconstruction) → 3D (HMD cross-attn)

---

## 8. 구현 현황

### Phase 1: 기본 구현 ✅
- ResNet backbone 유지
- Learnable joint queries + Self-attention
- Direct 3D head (x, y, z 동시 예측)

### Phase 2: Reconstruction Regularization ✅ (v2)
- Per-Joint Heatmap Decoder 추가
- Reconstruction loss로 2D 관절 정보를 joint tokens에 주입
- HMD Cross-Attention 후에 3D pose 예측

### Phase 3: 고해상도 확장 (TODO)
- layer3 사용 (8×8 → 16×16)
- 256 spatial tokens

---

## 9. 참고 문헌

- **ViTPose**: Simple Vision Transformer Baselines for Human Pose Estimation (NeurIPS 2022)
- **TokenPose**: Learning Keypoint Tokens for Human Pose Estimation (ICCV 2021)
- **DETR**: End-to-End Object Detection with Transformers (ECCV 2020)
