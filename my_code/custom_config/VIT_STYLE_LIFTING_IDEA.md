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

## 8. 버전별 구조 비교

### v1 (기존 - 별도 경로 Heatmap)
```
Backbone ─┬─→ Deconv → Heatmap → Heatmap Loss (별도)
          │
          └─→ Spatial Tokens + Joint Queries
                      ↓
              Self-Attention
                      ↓
              Joint Tokens
                      ↓
              HMD Cross-Attention
                      ↓
              3D Pose Head → 3D Loss
```
- Heatmap과 3D Lifting이 **독립적** (정보 공유 없음)

---

### v2 (No Reconstruction)
```
Backbone → Spatial Tokens + Joint Queries
                    ↓
            Self-Attention (spatial info 자동 전달)
                    ↓
            Joint Tokens
                    ↓
            HMD Cross-Attention
                    ↓
            3D Pose Head → 3D Loss만
```
- Heatmap 없음
- Self-Attention이 spatial 정보를 자연스럽게 전달한다고 가정

---

### v3 (Reconstruction Regularization) ⭐ 권장
```
Backbone → Spatial Tokens + Joint Queries
                    ↓
            Self-Attention
                    ↓
            Joint Tokens ◄── 공유 latent
                    │
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
Heatmap Decoder            HMD Cross-Attention
       │                         │
       ▼                         ▼
Recon Heatmap              3D Pose Head
       │                         │
       ▼                         ▼
Heatmap Loss ◄──────────────► 3D Loss
(regularizes tokens)
```
- Heatmap Reconstruction이 **Joint Tokens를 regularize**
- 2D 정보가 joint tokens에 강제 주입 → 더 나은 3D 예측

---

### 버전 비교 표

| | v1 | v2 | v3 |
|---|-----|-----|-----|
| Heatmap 경로 | 별도 (Deconv) | 없음 | **공유 (Decoder)** |
| Heatmap 역할 | Auxiliary only | - | **Token regularization** |
| 정보 흐름 | 독립적 | Self-Attn만 | **Recon + Self-Attn** |
| Config | `use_heatmap_recon=True` + deconv | `use_heatmap_recon=False` | `use_heatmap_recon=True` |
| 복잡도 | 높음 | 낮음 | 중간 |

### Smoke Test 결과 (1 epoch)

| | v2 | v3 |
|---|-----|-----|
| Full Body MPJPE | 303.59mm | **290.31mm** ✓ |
| Upper Body MPJPE | 206.22mm | **188.41mm** ✓ |
| Memory | 5891 MB | 7227 MB |

**결론**: v3 권장 (Heatmap reconstruction이 joint tokens 품질 향상)

---

## 9. 실행 방법

```bash
# v2 (No Reconstruction)
python tools/train.py my_code/custom_config/HMD_xregopose_vit_lifting_v2_full_config.py

# v3 (With Reconstruction) ⭐ 권장
python tools/train.py my_code/custom_config/HMD_xregopose_vit_lifting_v3_full_config.py
```

---

## 10. v3 전체 구조도 (상세)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT IMAGE [B, 3, 256, 256]                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKBONE (ResNet-101)                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│  │  layer1  │ → │  layer2  │ → │  layer3  │ → │  layer4  │                  │
│  │ 256ch    │   │ 512ch    │   │ 1024ch   │   │ 2048ch   │                  │
│  │ 64×64    │   │ 32×32    │   │ 16×16    │   │ 8×8      │                  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
│                                                     │                        │
│                                     backbone_feat [B, 2048, 8, 8]           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SPATIAL PROJECTION                                      │
│                                                                              │
│  backbone_feat [B, 2048, 8, 8]                                              │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────┐                                           │
│  │  Conv2d(2048 → 256, 1×1)    │  ← 채널 차원 압축 (2048 → 256)             │
│  │  BatchNorm2d(256)           │                                           │
│  │  ReLU                        │                                           │
│  └──────────────────────────────┘                                           │
│         │                                                                    │
│         ▼                                                                    │
│  spatial_feat [B, 256, 8, 8]                                                │
│         │                                                                    │
│         ▼  flatten(2) + transpose(1,2)                                      │
│  spatial_tokens [B, 64, 256]   ← 8×8=64개의 spatial tokens                  │
│         │                                                                    │
│         ▼  + PositionalEncoding2D (sinusoidal x,y)                          │
│         ▼  + spatial_type_embed (learnable)                                 │
│  spatial_tokens [B, 64, 256] (with position info)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │
          ┌─────────────────────────────┴─────────────────────────────┐
          │                                                           │
          ▼                                                           ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│   SPATIAL TOKENS          │                         │   LEARNABLE JOINT QUERIES │
│   [B, 64, 256]            │                         │                           │
│                           │                         │  nn.Parameter             │
│   8×8 grid positions      │                         │  [1, 16, 256]             │
│   with pos encoding       │                         │         │                 │
│                           │                         │         ▼ expand to B     │
│                           │                         │  + joint_type_embed       │
│                           │                         │  [B, 16, 256]             │
└───────────────────────────┘                         └───────────────────────────┘
          │                                                           │
          └───────────────────────┬───────────────────────────────────┘
                                  │ concat
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONCATENATED TOKENS                                  │
│                         [B, 80, 256]                                         │
│                                                                              │
│         [Spatial: 64 tokens] + [Joint: 16 tokens]                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION LAYERS (×4)                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  TransformerEncoderLayer (Pre-LayerNorm)                               │ │
│  │                                                                        │ │
│  │       ┌──────────────────────────────────────────────────┐             │ │
│  │       │  LayerNorm                                       │             │ │
│  │       │       ↓                                          │             │ │
│  │       │  MultiheadAttention(D=256, heads=8)              │             │ │
│  │       │  Q, K, V = all 80 tokens (bidirectional!)        │             │ │
│  │       │       ↓                                          │             │ │
│  │       │  Dropout + Residual                              │             │ │
│  │       └──────────────────────────────────────────────────┘             │ │
│  │                           ↓                                            │ │
│  │       ┌──────────────────────────────────────────────────┐             │ │
│  │       │  LayerNorm                                       │             │ │
│  │       │       ↓                                          │             │ │
│  │       │  MLP: Linear(256→1024) → GELU → Linear(1024→256) │             │ │
│  │       │       ↓                                          │             │ │
│  │       │  Dropout + Residual                              │             │ │
│  │       └──────────────────────────────────────────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                           × 4 layers                                         │
│                                                                              │
│  ★ Key: Spatial ↔ Joint 양방향 상호작용                                       │
│     - Joint → Spatial: "이 위치에 관절이 있나?"                               │
│     - Spatial → Joint: "이 관절의 feature 업데이트"                           │
│     - Joint ↔ Joint: "다른 관절과의 관계" (skeleton prior)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         LayerNorm(tokens)
                                  │
                                  ▼
                    Extract last 16 tokens (joint tokens)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOINT TOKENS [B, 16, 256]                            │
│                                                                              │
│  각 토큰 = 해당 관절의 latent representation                                  │
│  Self-Attention을 통해 spatial 정보 흡수                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                     │
                    │                                     │
    ┌───────────────┘                                     └───────────────┐
    │                                                                     │
    ▼                                                                     ▼
┌─────────────────────────────────────┐       ┌─────────────────────────────────────┐
│   HEATMAP DECODER (Reconstruction)  │       │                                     │
│   PerJointHeatmapDecoder            │       │   ★ 3D Prediction Path              │
│                                     │       │                                     │
│   joint_tokens [B, 16, 256]         │       └─────────────────────────────────────┘
│         │                           │                         │
│         ▼ reshape to [B*16, 256]    │                         │
│   ┌─────────────────────────┐       │                         ▼
│   │  MLP (shared)           │       │       ┌─────────────────────────────────────┐
│   │  Linear(256→256)        │       │       │         HMD CROSS-ATTENTION         │
│   │  LayerNorm + GELU       │       │       │                                     │
│   │  Linear(256→256)        │       │       │  Input:                             │
│   │  LayerNorm + GELU       │       │       │    Q: joint_tokens [B, 16, 256]     │
│   └─────────────────────────┘       │       │    HMD_info [B, 9]                  │
│         │                           │       │         │                           │
│         ▼ reshape [B*16, 256, 1, 1] │       │         ▼                           │
│   ┌─────────────────────────┐       │       │  ┌─────────────────────────┐        │
│   │  Progressive Upsample   │       │       │  │  HMD Embedding          │        │
│   │  1×1 → 3×3 → 6×6 →      │       │       │  │  Linear(9→512→768)     │        │
│   │  12×12 → 24×24 → 47×47  │       │       │  │  reshape to [B, 3, 256] │        │
│   │                         │       │       │  │  (head, R_hand, L_hand) │        │
│   │  ConvTranspose2d ×4     │       │       │  └─────────────────────────┘        │
│   │  + Bilinear Upsample    │       │       │         │                           │
│   │  + Conv2d (final)       │       │       │         ▼ hmd_tokens                │
│   └─────────────────────────┘       │       │  ┌─────────────────────────┐        │
│         │                           │       │  │  Cross-Attention        │        │
│         ▼                           │       │  │  (Pre-LayerNorm)        │        │
│   recon_heatmaps [B, 16, 47, 47]    │       │  │                         │        │
│         │                           │       │  │  Q: joint (normalized)  │        │
│         ▼                           │       │  │  K: hmd_tokens          │        │
│   ┌─────────────────────────┐       │       │  │  V: hmd_tokens          │        │
│   │  Heatmap Recon Loss     │       │       │  │                         │        │
│   │  (vs GT heatmaps)       │       │       │  │  + Dropout + Residual   │        │
│   │                         │       │       │  └─────────────────────────┘        │
│   │  ★ 2D 관절 위치 정보를   │       │       │         │                           │
│   │    joint tokens에       │       │       │         ▼                           │
│   │    강제 주입!           │       │       │   refined_joints [B, 16, 256]       │
│   └─────────────────────────┘       │       │                                     │
│         │                           │       │  ★ 3D depth reference 추가          │
│         ▼                           │       │    (머리/양손 위치 활용)             │
│   loss_heatmap_recon                │       └─────────────────────────────────────┘
│   (weight=500)                      │                         │
└─────────────────────────────────────┘                         │
                                                                ▼
                                      ┌─────────────────────────────────────────────┐
                                      │              3D POSE HEAD                    │
                                      │                                              │
                                      │  refined_joints [B, 16, 256]                │
                                      │         │                                    │
                                      │         ▼                                    │
                                      │  ┌──────────────────────────┐               │
                                      │  │  Linear(256 → 256)       │               │
                                      │  │  LayerNorm(256)          │               │
                                      │  │  GELU                    │               │
                                      │  │  Linear(256 → 128)       │               │
                                      │  │  GELU                    │               │
                                      │  │  Linear(128 → 3)         │               │
                                      │  └──────────────────────────┘               │
                                      │         │                                    │
                                      │         ▼                                    │
                                      │   pose_3d [B, 16, 3]                        │
                                      │   (x, y, z for each joint)                  │
                                      └─────────────────────────────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    LOSSES                                            │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │  3D Pose Losses:                                                               │ │
│  │    • loss_pose_l2norm     (weight=1.0)   - L2 distance                        │ │
│  │    • loss_cosine_similarity (weight=0.1) - Direction consistency              │ │
│  │    • loss_limb_length     (weight=0.25)  - Bone length consistency            │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │  HMD Loss:                                                                     │ │
│  │    • loss_hmd (weight=1.0) - Reconstructed HMD vs GT HMD                      │ │
│  │      pose_3d → _compute_hmd_from_pose() → hmd_recon [B, 9]                    │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Heatmap Reconstruction Loss (v3 핵심!):                                       │ │
│  │    • loss_heatmap_recon (weight=500)                                          │ │
│  │    • recon_heatmaps vs gt_heatmaps                                            │ │
│  │    ★ Joint tokens가 2D 관절 위치를 encoding하도록 강제                         │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT                                                │
│                                                                                      │
│   pose_3d [B, 16, 3]  →  16개 관절의 3D 좌표 (x, y, z)                              │
│                                                                                      │
│   Joint order:                                                                       │
│   [0] Head        [4] L_Wrist     [8] R_Knee      [12] L_Knee                       │
│   [1] Neck        [5] L_Elbow     [9] R_Ankle     [13] L_Ankle                      │
│   [2] R_Shoulder  [6] L_Shoulder  [10] R_Foot     [14] L_Foot                       │
│   [3] R_Elbow     [7] R_Wrist     [11] L_Hip      [15] Pelvis                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### v3 정보 흐름 요약

```
1. Backbone → Spatial Tokens [64, D]
2. + Learnable Joint Queries [16, D]
3. Self-Attention (양방향 상호작용)
4. Joint Tokens [16, D] ← 공유 latent representation
       │
       ├──→ Heatmap Decoder → Recon Loss (2D 정보 주입)
       │
       └──→ HMD Cross-Attention (3D depth 참조)
                   │
                   └──→ 3D Pose Head → [16, 3]
```

### 구조적 고려사항

**Heatmap Decoder 위치:**
- 현재: HMD Cross-Attention **이전**에 위치
- Heatmap은 pure visual task → HMD 없이도 가능해야 함
- 따라서 현재 위치가 적절

**Gradient 흐름:**
```
loss_heatmap_recon → heatmap_decoder → joint_tokens (HMD 이전)
                                              ↓
                                    Self-Attention layers

loss_3d → head_3d → joint_tokens (HMD 이후) → hmd_cross_attn
                                              ↓
                                    Self-Attention layers
```
- 두 loss 모두 Self-Attention을 통해 backbone까지 전파

---

## 11. v3 학습 안정성 분석 (Validation Spike 원인)

### 실험 결과 (10 epochs)

```
Epoch | Full Body | Upper Body | 비고
------|-----------|------------|------
  1   |   50.32   |   31.40    |
  2   |   53.72 ↑ |   28.27 ↓  | Full ↑, Upper ↓ (불일치)
  3   |   48.28 ↓ |   25.46 ↓  | LR decay (0.5x)
  4   |   45.34 ↓ |   23.49 ↓  | ★ Best
  5   |   47.13 ↑ |   24.76 ↑  | LR decay → 스파이크!
  6   |   49.12 ↑ |   26.22 ↑  | 스파이크 지속
  7   |   45.59 ↓ |   24.53 ↓  | LR decay → 회복
  8   |   47.11 ↑ |   25.49 ↑  | 다시 스파이크
  9   |   47.81   |   26.01    |
 10   |   47.37   |   24.56    |
```

### Walking Action의 극심한 변동

```
Epoch | Walking MPJPE | 변화
------|---------------|------
  1   |    52.38      |
  2   |    27.80      | ↓↓
  3   |    76.97      | ↑↑↑
  4   |    59.62      |
  5   |    25.28      | ↓↓
  6   |    93.95      | ↑↑↑ (최악)
  7   |    54.86      |
  8   |    22.99      | ↓↓ (최선)
  9   |    86.72      | ↑↑↑
 10   |    50.51      |
```
- **22mm ~ 94mm** 범위로 극심한 변동
- 하체 움직임이 많은 action에서 특히 불안정

---

### 스파이크 원인 분석

#### 1. LR Schedule 문제 (MultiStepLR)

```python
# 현재 설정
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5),
]
```

**LR 변화:**
```
Epoch 1-3: LR = 0.0005
Epoch 4:   LR = 0.00025 (3에서 decay) → Best!
Epoch 5:   LR = 0.000125 (5에서 decay) → 스파이크 시작
Epoch 7:   LR = 0.0000625 (7에서 decay)
```

**문제**: 급격한 LR 감소가 모델 불안정 유발

---

#### 2. HMD 정보의 한계

```
HMD 제공 정보:
  - 머리 위치 ✓
  - 오른손 위치 ✓
  - 왼손 위치 ✓

HMD 미제공:
  - 하체 (다리, 발) ✗
```

**결과:**
- Upper Body: HMD 정보로 안정적 예측
- Lower Body: 순수 visual feature 의존 → 불안정
- Walking 같은 동적 action에서 하체 예측 실패

---

#### 3. Heatmap Reconstruction Loss Weight

```python
loss_heatmap_recon=dict(loss_weight=500, ...)  # 다른 loss 대비 매우 높음
```

| Loss | Weight | 비율 |
|------|--------|------|
| loss_pose_l2norm | 1.0 | 1x |
| loss_cosine_similarity | 0.1 | 0.1x |
| loss_limb_length | 0.25 | 0.25x |
| loss_hmd | 1.0 | 1x |
| **loss_heatmap_recon** | **500** | **500x** |

**문제:**
- Reconstruction 품질 변동 → total loss 큰 변동
- 2D heatmap 학습이 dominant → 3D prediction 방해 가능

---

#### 4. PerJointHeatmapDecoder 구조

```python
# 1×1 → 47×47 로 급격한 upsampling (2209배)
self.upsample = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 3, 1, 0),  # 1→3
    nn.BatchNorm2d(128),  # ← 작은 spatial에서 불안정
    nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 3→6
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 6→12
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 64, 4, 2, 1),    # 12→24
    nn.BatchNorm2d(64),
    nn.Upsample(size=(47, 47)),              # 24→47
    ...
)
```

**문제:**
- 1×1에서 시작하는 극단적 upsampling
- 작은 spatial size에서 BatchNorm 통계 불안정
- ConvTranspose의 checkerboard artifacts

---

#### 5. Full Body vs Upper Body 불일치

Epoch 2 예시:
```
Full Body:  50.32 → 53.72 (악화)
Upper Body: 31.40 → 28.27 (개선)
```

**의미:**
- 손/팔은 개선되는데 다리가 악화
- HMD Cross-Attention이 상체에만 효과적

---

### 해결 방안

#### Option 1: LR Schedule 수정 (권장)

```python
# CosineAnnealingLR로 부드러운 감소
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

#### Option 2: Heatmap Recon Loss Weight 감소

```python
# 500 → 100~200으로 감소
loss_heatmap_recon=dict(loss_weight=200, ...)
```

#### Option 3: Lower Body Weight 증가

```python
# 하체 관절에 더 높은 weight
keypoint_weights = [
    1.0,  # Head
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Upper body
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  # Lower body (1.5x)
]
```

#### Option 4: BatchNorm → GroupNorm

```python
# PerJointHeatmapDecoder에서
nn.BatchNorm2d(128)  # 불안정
    ↓
nn.GroupNorm(8, 128)  # 더 안정적
```

---

### 권장 실험 순서

1. **Option 1 (LR Schedule)** 먼저 적용 - 가장 간단, 효과 클 것으로 예상
2. 효과 없으면 **Option 2 (Loss Weight)** 추가
3. 여전히 불안정하면 **Option 4 (GroupNorm)** 적용

---

## 12. v4: CosineAnnealingLR 적용

### v3 대비 변경점

```python
# v3 (MultiStepLR - 급격한 decay)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5),
]

# v4 (CosineAnnealingLR - 부드러운 decay)
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

### LR 변화 비교

```
v3 MultiStepLR:
  Epoch 1-3: 0.0005
  Epoch 4:   0.00025 (갑자기 50% 감소!)
  Epoch 5:   0.000125 (갑자기 50% 감소!)
  → 급격한 변화로 validation spike 유발

v4 CosineAnnealingLR:
  Epoch 0:  0.0005
  Epoch 5:  ~0.00025 (부드럽게 감소)
  Epoch 10: 0.000001
  → 부드러운 변화로 안정적 학습
```

### Smoke Test 결과 (2 epochs)

| Version | Epoch 1 Full Body | Epoch 2 Full Body |
|---------|-------------------|-------------------|
| v3 (MultiStepLR) | 290.31mm | - |
| **v4 (CosineAnnealingLR)** | **281.73mm** | - |

**결론**: v4가 더 안정적인 수렴 시작

---

## 13. v5: Hybrid Attention 구조

### v4 대비 핵심 변경점

1. **Attention 구조 변경**
   - v4: Self-Attention [80×80] (Spatial 64 + Joint 16)
   - v5: Cross-Attention [16×64] + Self-Attention [16×16]

2. **역할 분리**
   - Cross-Attention: 관절이 이미지에서 위치 찾기
   - Self-Attention: 관절 간 skeleton 관계 학습

3. **계산 효율**
   - 6400 attention weights → 1280 (5배 감소)

4. **Gradient Scaling**
   - Heatmap gradient를 0.1배로 스케일링
   - 3D prediction 학습에 집중

### v5 구조도

```
┌─────────────────────────────────────────────────────────────┐
│  Backbone feat [2048, 8, 8]                                 │
│         ↓                                                   │
│  Spatial Tokens [64, D]                                     │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Stage 1: Cross-Attention (J → S)       │                │
│  │  Q: Joint Queries [16, D]               │                │
│  │  K/V: Spatial Tokens [64, D]            │  [16×64]       │
│  │  → 각 관절이 이미지에서 위치 찾기         │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Stage 2: Self-Attention (J → J) × N    │                │
│  │  Q=K=V: Joint Tokens [16, D]            │  [16×16]       │
│  │  → 관절 간 skeleton 관계 학습            │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Joint Tokens [16, D]                                       │
│         │                                                   │
│         ├── Heatmap Decoder (gradient scaled by 0.1)        │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Stage 3: HMD Cross-Attention           │                │
│  │  Q: Joint Tokens, K/V: HMD Tokens       │  [16×3]        │
│  │  → 3D depth reference 추가               │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│    3D Pose Head → [16, 3]                                   │
└─────────────────────────────────────────────────────────────┘
```

### v4 vs v5 Attention 비교

```
v4 Self-Attention [80×80]:
┌─────────────────────────────────────────────────────────────┐
│                    Q = K = V (80 tokens)                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  S→S (64×64)  │  S→J (64×16)  │  ← 불필요한 계산         │
│  │  J→S (16×64)  │  J→J (16×16)  │  ← 필요한 계산           │
│  └────────────────────────────────────────────────────────┘ │
│  Total: 6400 attention weights                              │
│  문제: S→S, S→J는 3D prediction에 불필요                     │
└─────────────────────────────────────────────────────────────┘

v5 Hybrid Attention:
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Cross-Attention (J → S) [16×64]                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Q: Joint [16, D]                                      │ │
│  │  K, V: Spatial [64, D]                                 │ │
│  │  → 관절별로 relevant spatial 위치에 attend              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Stage 2: Self-Attention (J → J) [16×16]                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Q = K = V: Joint [16, D]                              │ │
│  │  → 관절 간 skeleton 관계 학습                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Total: 1024 + 256 = 1280 attention weights                 │
│  장점: 필요한 attention만 집중                               │
└─────────────────────────────────────────────────────────────┘
```

### Gradient Scaling 구현

```python
class GradientScale(torch.autograd.Function):
    """Forward는 정상, Backward는 scale배."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

# 사용
if self.use_heatmap_recon:
    # Heatmap gradient = 0.1배 (3D prediction 학습에 집중)
    scaled_tokens = gradient_scale(joint_tokens, 0.1)
    recon_heatmaps = self.heatmap_decoder(scaled_tokens)
```

### Smoke Test 결과 (2 epochs)

| Version | Epoch 1 Full Body | Epoch 2 Full Body |
|---------|-------------------|-------------------|
| v4 (Self-Attn) | 290.31mm | 281.73mm |
| **v5 (Hybrid)** | **272.93mm** | **204.41mm** |

**결론**: v5가 더 빠른 수렴을 보여줌

---

## 14. Joint Tokens에 Depth 정보 주입 방법

### 현재 문제

```
Joint Queries (learnable)  →  2D spatial attention만으로 3D 학습
                              ↓
                          Depth cues가 implicit하게만 전달됨
```

### 제안하는 해결책

#### Option 1: Depth-Aware Joint Initialization

```python
# 기존: 무작위 초기화
joint_queries = Parameter(randn(1, 16, D))

# 개선: skeleton depth prior 포함
joint_queries = Parameter(
    concat([
        learnable_2d,      # [16, D//2] - 2D 위치 학습
        depth_prior        # [16, D//2] - skeleton 기반 depth prior
    ])
)
```

- 인체 skeleton의 평균 depth 순서를 encoding
  - head → shoulder → elbow → wrist (순차적)
- 관절별 typical depth range를 prior로 제공

#### Option 2: Multi-Scale Spatial Features

```
현재: backbone feat [8×8] → spatial tokens [64]

개선: multi-scale features
    - [8×8]   → 64 tokens (coarse, depth context)
    - [16×16] → 256 tokens (fine, location)
    Cross-Attention: Joint queries → Multi-scale tokens
```

- 작은 feature map: 넓은 receptive field → depth context 포함
- 큰 feature map: precise localization

#### Option 3: Early HMD Injection (권장)

```
현재:
    Cross-Attn (J→S) → Self-Attn (J→J) → HMD Cross-Attn (맨 마지막)

개선:
    Cross-Attn (J→S) → [HMD Injection] → Self-Attn (J→J)
                              ↑
                        초기에 depth anchor 제공
```

- HMD가 제공하는 head/hand depth를 early stage에서 주입
- 다른 관절들이 이 anchor를 기준으로 relative depth 학습
- **가장 간단하게 구현 가능, HMD 활용 극대화**

#### Option 4: Explicit Depth Token

```python
# Joint Tokens [16, D] + Depth Token [1, D]

depth_token = nn.Parameter(torch.randn(1, 1, D))  # Global depth context

# Self-Attention에서
tokens = concat([joint_tokens, depth_token])  # [17, D]
tokens = self_attention(tokens)
joint_tokens = tokens[:, :16]  # Depth token 제외
```

- Global depth context를 담당하는 별도 token
- 모든 joint가 이 token과 attention
- Self-Attn에서 depth reference 역할

### 권장 구현 우선순위

1. **Option 3 (Early HMD Injection)** - 가장 간단, v6에서 구현 예정
2. **Option 1 (Depth-Aware Init)** - skeleton prior 활용
3. **Option 2 (Multi-Scale)** - 계산량 증가하지만 효과적

---

## 15. 버전별 요약

| Version | 구조 | LR Schedule | Attention | 특징 |
|---------|------|-------------|-----------|------|
| v1 | 별도 Heatmap + Lifting | - | - | Heatmap/3D 독립 |
| v2 | Self-Attention only | MultiStep | Self [80×80] | No Recon |
| v3 | Self-Attention + Recon | MultiStep | Self [80×80] | Recon regularization |
| **v4** | Self-Attention + Recon | **CosineAnnealing** | Self [80×80] | 안정적 LR decay |
| **v5** | **Hybrid Attention** + Recon | CosineAnnealing | **Cross [16×64] + Self [16×16]** | 역할 분리, Grad Scale |
| v6 (예정) | Hybrid + Early HMD | CosineAnnealing | Cross + Self | Depth prior 강화 |

---

## 16. 향후 확장 (TODO)

### Phase 3: 고해상도 확장
- layer3 사용 (8×8 → 16×16)
- 256 spatial tokens

### v6: Early HMD Injection
- HMD Cross-Attention을 Self-Attention 이전으로 이동
- Depth anchor를 초기에 제공

---

## 17. 참고 문헌

- **ViTPose**: Simple Vision Transformer Baselines for Human Pose Estimation (NeurIPS 2022)
- **TokenPose**: Learning Keypoint Tokens for Human Pose Estimation (ICCV 2021)
- **DETR**: End-to-End Object Detection with Transformers (ECCV 2020)
