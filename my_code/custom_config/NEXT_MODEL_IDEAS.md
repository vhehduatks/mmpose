# 다음 모델 아이디어 (2026-01-27)

## 문제 분석

### 핵심 발견

| 모델 | Full Body | Upper Body | Lower Body |
|------|-----------|------------|------------|
| **Baseline** | **41.37mm** | 29.42mm | **53.31mm** |
| **ViT v3** | 45.34mm | **23.49mm** ⭐ | 67.19mm |

**ViT v3의 역설:**
- Upper Body: Baseline보다 **-5.93mm 좋음** (최고!)
- Lower Body: Baseline보다 **+13.88mm 나쁨**
- Full Body 차이(+3.97mm)의 대부분이 Lower Body에서 발생

### 원인 분석

```
HMD 정보 구성:
  ✓ Head position (3D)    → Upper Body에 유리
  ✓ Right hand (3D)       → Upper Body에 유리
  ✓ Left hand (3D)        → Upper Body에 유리
  ✗ Lower body 정보 없음   → Lower Body에 불리

Baseline의 강점:
  - HeatmapEncoder가 전체 body의 2D 정보를 균등하게 Z에 encoding
  - Z[64]가 모든 관절에 대한 balanced representation
  - HMD는 보조 역할만 (Z + HMD concat)

ViT v3의 약점:
  - HMD Cross-Attention이 상체에 편향된 depth reference 제공
  - Joint Tokens가 HMD 기준으로 depth 학습
  - Lower Body는 Self-Attention만으로 depth 추정 → HMD anchor 없이 불안정
```

---

## Option 1: Upper-Lower Decoupled Network (권장) ⭐

### 핵심 아이디어
- Upper Body: ViT v3 스타일 (HMD 활용, 23.49mm 달성)
- Lower Body: Baseline 스타일 (HMD 없이, Z vector 기반)
- 각 부위에 최적화된 구조를 분리하여 사용

### 구조

```
                    Backbone feat [2048, 8, 8]
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  Upper Body Branch  │       │  Lower Body Branch  │
    │  (ViT v3 style)     │       │  (Baseline style)   │
    │                     │       │                     │
    │  - Spatial Tokens   │       │  - Deconv           │
    │  - Joint Queries[8] │       │  - Heatmap[8]       │
    │  - Self-Attention   │       │  - HeatmapEncoder   │
    │  - HMD Cross-Attn   │       │  - Z vector [64]    │
    │  - Heatmap Recon    │       │  - PoseDecoder      │
    └─────────────────────┘       └─────────────────────┘
              │                               │
              │  Upper pose [8, 3]            │  Lower pose [8, 3]
              │                               │
              └───────────────┬───────────────┘
                              │
                        Concat → [16, 3]
                              │
                   (Optional) Refinement Layer
                              │
                        3D Pose [16, 3]
```

### 관절 분할

**Upper Body (8 joints):**
- 0: Head
- 1: Neck
- 2: R_Shoulder
- 3: R_Elbow
- 4: L_Wrist (HMD)
- 5: L_Elbow
- 6: L_Shoulder
- 7: R_Wrist (HMD)

**Lower Body (8 joints):**
- 8: R_Knee
- 9: R_Ankle
- 10: R_Foot
- 11: L_Hip
- 12: L_Knee
- 13: L_Ankle
- 14: L_Foot
- 15: Pelvis

### 예상 결과

```
Upper Body: 23.49mm (ViT v3 수준)
Lower Body: 53.31mm (Baseline 수준)
Full Body:  (23.49 * 8 + 53.31 * 8) / 16 = 38.40mm

→ Baseline(41.37mm)보다 ~3mm 개선 가능!
```

### 장점
- 각 부위에 최적화된 구조 사용
- HMD 정보가 있는 Upper Body에만 HMD Cross-Attention 적용
- Lower Body는 Baseline의 검증된 구조 재사용

### 단점
- 모델 복잡도 증가 (두 개의 branch)
- Upper/Lower 경계 관절의 일관성 문제 가능

---

## Option 2: Hierarchical Pose Estimation (v2 - Cross-Attention 기반)

### 핵심 아이디어
- Stage 1: Upper Body를 먼저 예측 (anchor)
- Stage 2: Upper pose + 이미지를 조건으로 Lower Body 예측
- **Cross-Attention으로 2D 위치 학습** (Self-Attention 중복 제거)
- **Heatmap Recon Loss로 2D supervision** 보장

### 구조

```
                    Backbone feat [2048, 8, 8]
                              │
                              ▼
                    Spatial Tokens [64, D]  ← 1번만 생성 (공유)
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  Stage 1: Upper Body         │ │  Stage 2: Lower Body         │
│                              │ │                              │
│  Upper Queries [8, D]        │ │  Lower Queries [8, D]        │
│           │                  │ │           │                  │
│           ▼                  │ │           ▼                  │
│  Cross-Attn (2D):            │ │  Cross-Attn (2D):            │
│    Q: Upper Queries          │ │    Q: Lower Queries          │
│    K/V: Spatial Tokens       │ │    K/V: Spatial Tokens       │
│           │                  │ │           │                  │
│           ▼                  │ │           ▼                  │
│  Upper Tokens [8, D]─────────┼─┼─→ Cross-Attn (Upper):       │
│           │                  │ │      Q: Lower Tokens         │
│           ├─→ Heatmap Recon  │ │      K/V: Fused Upper        │
│           │   (2D loss) ✅   │ │           │                  │
│           ▼                  │ │           ├─→ Heatmap Recon  │
│  HMD Cross-Attn              │ │           │   (2D loss) ✅   │
│           │                  │ │           ▼                  │
│           ▼                  │ │  3D Head                     │
│  3D Head → Upper [8,3]───────┼─┘           │                  │
│           │                  │              ▼                  │
│           ▼                  │   Lower Pose [8, 3]           │
│  Pose Embed [8, D]───────────┼─→ (Fusion Layer로 전달)       │
│                              │                                │
└──────────────────────────────┘ └─────────────────────────────┘
                                              │
                                              ▼
                              Concat → Full Pose [16, 3]
```

### Fusion Layer (Upper 정보 결합)

```python
# Upper Joint Tokens [8, D] - 이미지 특징
# Upper Pose Embedding [8, D] - 3D 좌표 정보
fused = self.fusion_linear(
    torch.cat([upper_joint_tokens, upper_pose_embedding], dim=-1)
)  # [8, 2D] → [8, D]
```

### Loss 구성

| Loss | 대상 | 역할 |
|------|------|------|
| `loss_upper_heatmap_recon` | Upper Heatmap [8] | **Upper 2D 위치 학습** |
| `loss_lower_heatmap_recon` | Lower Heatmap [8] | **Lower 2D 위치 학습** |
| `loss_pose_l2norm` | Full Pose [16, 3] | 3D 좌표 |
| `loss_cosine_similarity` | Full Pose | 방향 |
| `loss_limb_length` | Full Pose | 뼈 길이 |
| `loss_hmd` | HMD reconstruction | HMD 일관성 |

### 장점
- **Spatial Tokens 공유**: 중복 연산 제거
- **Cross-Attention으로 2D 학습**: Self-Attention [72×72] → Cross-Attention [8×64]
- **Heatmap Recon Loss**: 2D supervision 보장
- **Hierarchical 구조 유지**: Lower가 Upper 결과 참조
- Lower Body가 두 가지 정보 모두 활용:
  1. Spatial Tokens → 이미지 2D 위치
  2. Fused Upper → Upper Body와의 3D 관계

### 단점
- Upper Body 에러가 Lower Body로 전파 가능
- Stage 간 gradient 흐름 설계 필요 (detach 여부)

---

## Option 3: Symmetric Prior Enhancement

### 핵심 아이디어
- 좌/우 관절의 대칭성을 prior로 활용
- Standing pose에서 좌우 depth가 유사해야 함

### 구조

```
Joint Tokens [16, D]
       │
       ▼
┌─────────────────────────────────────┐
│  Symmetric Attention Layer          │
│                                     │
│  Symmetric pairs:                   │
│    - L_Shoulder ↔ R_Shoulder       │
│    - L_Elbow ↔ R_Elbow             │
│    - L_Wrist ↔ R_Wrist             │
│    - L_Hip ↔ R_Hip                 │
│    - L_Knee ↔ R_Knee               │
│    - L_Ankle ↔ R_Ankle             │
│    - L_Foot ↔ R_Foot               │
│                                     │
│  Attention: Q=Left, K/V=Right      │
│           + Q=Right, K/V=Left      │
└─────────────────────────────────────┘
       │
       ▼
  Refined Tokens
       │
       ▼
  3D Pose Head
```

### Symmetric Loss

```python
def symmetric_depth_loss(pose_3d):
    pairs = [(2, 6), (3, 5), (4, 7), (8, 12), (9, 13), (10, 14)]  # L-R pairs
    loss = 0
    for l_idx, r_idx in pairs:
        # Depth difference penalty
        depth_diff = torch.abs(pose_3d[:, l_idx, 2] - pose_3d[:, r_idx, 2])
        loss += depth_diff.mean()
    return loss
```

### 장점
- 구현 간단
- 기존 모델에 추가 가능
- 좌우 일관성 향상

### 단점
- 비대칭 포즈 (한 발 들기 등)에서 penalty
- Action에 따라 dynamic weight 필요

---

## Option 4: Lower Body Pelvis Proxy

### 핵심 아이디어
- HMD가 없는 Lower Body에 pseudo-anchor 제공
- Upper Body 결과로부터 Pelvis 위치를 추정
- Pelvis를 Lower Body의 "HMD" 역할로 사용

### 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Upper Body + Pelvis                               │
│                                                             │
│  Backbone feat → ViT v3 style                              │
│        │                                                    │
│        ↓                                                    │
│  Joint Queries [9] (8 upper + 1 pelvis)                    │
│        │                                                    │
│        ↓                                                    │
│  Self-Attention + HMD Cross-Attention                      │
│        │                                                    │
│        ↓                                                    │
│  Upper pose [8, 3] + Pelvis [1, 3]                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Lower Body (anchored by Pelvis)                   │
│                                                             │
│  Lower Joint Queries [7] (excluding pelvis)                │
│        │                                                    │
│        ↓                                                    │
│  Pelvis Cross-Attention:                                   │
│    Q: Lower Queries [7]                                    │
│    K/V: Pelvis token [1]                                   │
│        │                                                    │
│        ↓                                                    │
│  Lower pose [7, 3]                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Full pose [16, 3]
```

### Pelvis 추정 방법

```python
# Option A: Learnable query
pelvis_query = nn.Parameter(torch.randn(1, 1, D))

# Option B: Upper pose에서 추정
def estimate_pelvis(upper_pose):
    # Pelvis ≈ midpoint of shoulders + offset
    l_shoulder = upper_pose[:, 6]  # L_Shoulder
    r_shoulder = upper_pose[:, 2]  # R_Shoulder
    neck = upper_pose[:, 1]        # Neck

    shoulder_mid = (l_shoulder + r_shoulder) / 2
    # Pelvis is roughly below neck
    pelvis = shoulder_mid + (shoulder_mid - neck) * 1.5
    return pelvis
```

### 장점
- HMD 없이도 Lower Body에 depth anchor 제공
- Pelvis가 자연스러운 body center 역할

### 단점
- Pelvis 추정 오류가 Lower Body로 전파
- 2-stage 학습 복잡도

---

## 구현 우선순위

| 순위 | Option | 예상 효과 | 복잡도 | 리스크 | 상태 |
|------|--------|----------|--------|--------|------|
| 1 | **Option 1: Decoupled** | ~38mm | 중간 | 낮음 | ✅ 구현 완료 |
| 2 | **Option 2: Hierarchical** | ~39mm | 중간 | 중간 | ✅ 구현 완료 |
| 3 | Option 3: Symmetric | +1~2mm | 낮음 | 낮음 | 대기 |
| 4 | Option 4: Pelvis Proxy | ~40mm | 중간 | 중간 | 대기 |
| 5 | **Option 5: ViT v6 (SPT+LSA)** | ~43mm | 낮음 | 낮음 | ✅ 구현 완료 |

---

## Option 1 구현 완료 (2026-01-27)

### 구현 파일

| 파일 | 용도 |
|------|------|
| `custom_egopose_decoupled_head.py` | Head 구현체 |
| `HMD_xregopose_decoupled_small_config.py` | Smoke test config |
| `HMD_xregopose_decoupled_full_config.py` | Full training config |

### Smoke Test 결과 (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 637.00mm | 173.13mm | 1100.86mm |
| 2 | 531.01mm | 124.07mm | 937.95mm |

✅ 모델이 정상적으로 학습됨
✅ Upper/Lower body 각각 개선 확인
✅ 모든 loss 컴포넌트 작동 확인

### 다음 단계

1. Full training (10 epochs)
2. 결과 분석
3. 필요시 Option 3-4 구현

---

## Option 2 구현 완료 (2026-01-27)

### 구현 파일

| 파일 | 용도 |
|------|------|
| `custom_egopose_hierarchical_head.py` | Head 구현체 |
| `HMD_xregopose_hierarchical_small_config.py` | Smoke test config |
| `HMD_xregopose_hierarchical_full_config.py` | Full training config |

### Smoke Test 결과 (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 264.39mm | 177.70mm | 351.09mm |
| 2 | 194.68mm | 137.30mm | 252.06mm |

✅ 모델이 정상적으로 학습됨
✅ Hierarchical 구조 (Lower가 Upper 참조) 작동 확인
✅ Cross-Attention 기반 2D 학습 + Heatmap Recon Loss 작동 확인

---

---

## Option 5: ViT Lifting v6 (Small Dataset Optimized) ⭐ 신규

### 배경

**문제점**: EgoPose 데이터셋(210K)은 ViT가 효과적으로 학습하기엔 매우 작음
- ViT 최소 요구량: 14M+ (ImageNet-21k)
- EgoPose: 210K (요구량의 **1.5%**)
- ViT v3의 validation spike와 학습 불안정의 근본 원인

**해결책**: Small Dataset 전용 ViT 기법 적용
- **SPT (Shifted Patch Tokenization)**: Locality inductive bias 주입
- **LSA (Locality Self-Attention)**: Learnable temperature + Diagonal masking
- **모델 축소**: Overfitting 방지

### 핵심 기법

#### 1. Shifted Patch Tokenization (SPT)

```python
# 기존: 단순 projection
spatial_tokens = self.proj(backbone_feat)  # [B, D, 8, 8]

# SPT: 5방향 shift로 locality 강화
x_left  = F.pad(x, (1, 0, 0, 0))[:, :, :, :W]   # 왼쪽 shift
x_right = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]   # 오른쪽 shift
x_up    = F.pad(x, (0, 0, 1, 0))[:, :, :H, :]   # 위 shift
x_down  = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]   # 아래 shift

x_concat = torch.cat([x, x_left, x_right, x_up, x_down], dim=1)  # [B, 5C, H, W]
spatial_tokens = self.proj(x_concat)  # Receptive field 5배 확장
```

**효과**: 이웃 픽셀 정보를 token에 직접 주입 → CNN의 locality bias 모방

#### 2. Locality Self-Attention (LSA)

```python
class LocalitySelfAttention:
    def __init__(self):
        # 핵심 1: Learnable temperature (초기값 낮게 → sharp attention)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # Attention scores
        attn = (Q @ K.T) / sqrt(D)
        attn = attn / self.temperature.clamp(min=0.1)  # Sharp attention

        # 핵심 2: Diagonal masking (self-relation 제거)
        diag_mask = torch.eye(N).bool()
        attn = attn.masked_fill(diag_mask, float('-inf'))

        return softmax(attn) @ V
```

**효과**:
- Learnable temperature: 초기에 sharp attention → 이웃에 집중
- Diagonal masking: 자기 자신 무시 → 주변 정보만 활용

#### 3. 모델 축소

| 항목 | v3 (기존) | v6 (축소) | 감소율 |
|------|----------|----------|--------|
| embed_dim | 256 | 128 | 50% |
| num_heads | 8 | 4 | 50% |
| num_layers | 4 | 2 | 50% |
| mlp_ratio | 4.0 | 2.0 | 50% |
| dropout | 0.1 | 0.2 | +100% |
| **총 params** | ~5M | ~1.2M | **76%↓** |

### 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Backbone feat [2048, 8, 8]                                 │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  SPT (Shifted Patch Tokenization)       │  ← NEW         │
│  │  5방향 shift: [2048,8,8] → [10240,8,8]  │                │
│  │  → proj → [128, 8, 8]                   │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Depth-wise Conv Embedding              │  ← NEW         │
│  │  DWConv(3×3) → BN → GELU               │                │
│  │  Local feature enhancement              │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Spatial Tokens [64, 128] + Joint Queries [16, 128]         │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  LSA (Locality Self-Attention) × 2      │  ← NEW         │
│  │  - Learnable temperature (init=0.5)     │                │
│  │  - Diagonal masking (self-relation 제거)│                │
│  │  - mlp_ratio=2.0 (축소)                 │                │
│  └─────────────────────────────────────────┘                │
│         ↓                                                   │
│  Joint Tokens [16, 128]                                     │
│         │                                                   │
│         ├── Heatmap Decoder (Reconstruction)                │
│         │   (2D 정보 강제 주입)                              │
│         ↓                                                   │
│  HMD Cross-Attention [16×3]                                 │
│         ↓                                                   │
│  3D Pose Head → [16, 3]                                     │
└─────────────────────────────────────────────────────────────┘
```

### v3 대비 변경 요약

| 항목 | v3 | v6 | 효과 |
|------|-----|-----|------|
| Tokenization | Linear proj | **SPT** | Locality bias ↑ |
| Self-Attention | Standard | **LSA** | Sharp local attention |
| embed_dim | 256 | **128** | Overfitting ↓ |
| num_layers | 4 | **2** | 파라미터 ↓ |
| mlp_ratio | 4.0 | **2.0** | 파라미터 ↓ |
| dropout | 0.1 | **0.2** | Regularization ↑ |
| DWConv | ❌ | **✅** | Local features |

### 예상 효과

```
v3 문제점:
- Epoch 2, 5 validation spike
- 210K 데이터로 256-dim, 4-layer 학습 → overfitting
- Global attention이 small dataset에서 불안정

v6 기대:
- SPT로 locality bias → 초기 학습 안정화
- LSA로 sharp local attention → 이웃 관절 관계 학습
- 모델 축소로 overfitting 방지
- validation spike 감소 → 일관된 성능
```

### 참고 논문

- [Vision Transformer for Small-Size Datasets (AAAI 2022)](https://arxiv.org/abs/2112.13492)
- [Depth-Wise Convolutions in ViTs (Neural Networks 2024)](https://www.sciencedirect.com/science/article/pii/S0925231224017697)

---

## Option 5 구현 완료 (2026-01-27)

### 구현 파일

| 파일 | 용도 |
|------|------|
| `custom_egopose_vit_lifting_head_v6.py` | Head 구현체 (SPT + LSA + 축소 모델) |
| `HMD_xregopose_vit_lifting_v6_small_config.py` | Smoke test config |
| `HMD_xregopose_vit_lifting_v6_full_config.py` | Full training config |

### Smoke Test 결과 (2 epochs, small dataset)

| Epoch | Full Body | Upper Body | Lower Body | 개선율 |
|-------|-----------|------------|------------|--------|
| 1 | 248.77mm | 185.22mm | 312.32mm | - |
| 2 | 207.42mm | 157.40mm | 257.44mm | -16.6% |

✅ 모델이 정상적으로 학습됨
✅ SPT (Shifted Patch Tokenization) 작동 확인
✅ LSA (Locality Self-Attention) 작동 확인
✅ Depth-wise Conv Embedding 작동 확인
✅ 축소된 모델 (embed_dim=128, num_layers=2) 작동 확인

### v3 vs v6 Smoke Test 비교 (참고)

| 항목 | v3 | v6 | 비고 |
|------|-----|-----|------|
| embed_dim | 256 | 128 | 50% 축소 |
| num_layers | 4 | 2 | 50% 축소 |
| Params (추정) | ~5M | ~1.2M | 76% 감소 |
| GPU Memory | ~7GB | ~6GB | ~14% 절감 |

### 다음 단계

1. Full training (10 epochs)
2. v3과 비교 (validation spike 감소 확인)
3. 결과가 좋으면 SPT/LSA 개별 ablation study

---

## 참고: 관절 인덱스 (xRegopose)

```
Upper Body (8):
  0: Head
  1: Neck
  2: R_Shoulder
  3: R_Elbow
  4: L_Wrist
  5: L_Elbow
  6: L_Shoulder
  7: R_Wrist

Lower Body (8):
  8: R_Knee
  9: R_Ankle
  10: R_Foot
  11: L_Hip
  12: L_Knee
  13: L_Ankle
  14: L_Foot
  15: Pelvis
```
