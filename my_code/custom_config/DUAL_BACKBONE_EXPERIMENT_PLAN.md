# Dual Backbone 개선 실험 계획

## 목표

**Single COCO (42.07mm)보다 나은 성능을 Dual Backbone mutual learning으로 달성하기.**

현재 문제: Dual COCO+MPII (44.91mm)가 Single COCO (42.07mm)보다 2.84mm 나쁨

---

## 베이스라인 결과 (Updated: 2026-01-20)

### Wandb 실험 결과

**mmpose_xregopose_single_coco:**
| Run | State | Full Body | Upper Body | Lower Body |
|-----|-------|-----------|------------|------------|
| fresh-haze-6 | running | **42.07mm** | 29.15mm | 55.00mm |

**mmpose_xregopose_coco_mpii:**
| Run | State | Full Body | Upper Body | Lower Body |
|-----|-------|-----------|------------|------------|
| full셋 test로 val | finished | 44.91mm | 30.56mm | 59.26mm |

### 핵심 비교

| Model | Full Body MPJPE | 차이 | 비고 |
|-------|-----------------|------|------|
| **Single COCO** | **42.07mm** | - | 🏆 현재 최고 |
| Dual COCO+MPII | 44.91mm | +2.84mm | Dual이 약간 나쁨 |

---

## 현재 모델 구조

### Single COCO (42.07mm)

**Config**: `HMD_xregopose_single_coco_full_config.py`

```
┌─────────────────────────────────────────────────────────────┐
│  TopdownPoseEstimator                                       │
├─────────────────────────────────────────────────────────────┤
│  Backbone: ResNet-101 (COCO pretrained)                     │
│       ↓                                                     │
│  Head: CustomxRegoposeBaselinel1                            │
│       ├── Deconv → Heatmap [16, 47, 47]                     │
│       ├── Encoder → Z [64]                                  │
│       ├── + HMD info [9→64]                                 │
│       ├── Pose Decoder → 3D Pose [16, 3]                    │
│       └── Heatmap Decoder → Recon Heatmap                   │
└─────────────────────────────────────────────────────────────┘
```

**Loss 구성**:
| Loss | Weight | 설명 |
|------|--------|------|
| loss_kpt | 1000 | Main heatmap MSE |
| loss_heatmap_recon | 250 | Heatmap reconstruction |
| loss_pose_l2norm | 1.0 | 3D pose L2 |
| loss_cosine_similarity | 0.1 | Cosine sim |
| loss_limb_length | 0.25 | Limb length |
| loss_hmd | 1.0 | HMD reconstruction |

---

### Dual COCO+MPII (44.91mm)

**Config**: `HMD_xregopose_h5cache_coco_mpii_config.py`

```
┌─────────────────────────────────────────────────────────────┐
│  Custom_TopdownPoseEstimator                                │
├─────────────────────────────────────────────────────────────┤
│  Backbone1: ResNet-101 (COCO pretrained)                    │
│  Backbone2: ResNet-101 (MPII pretrained)                    │
│       ↓                    ↓                                │
│  feat1 [2048,8,8]     feat2 [2048,8,8]                      │
│       ↓                    ↓                                │
│  Deconv1 → Heatmap1   Deconv2 → Heatmap2                    │
│       ↓                    ↓                                │
│       └────── MSE Loss ────┘  ← loss_backbone_latant        │
│       ↓                    ↓                                │
│  Main path            Sub path (GT supervision)             │
│       ↓                                                     │
│  Encoder → Z → Pose Decoder → 3D Pose                       │
└─────────────────────────────────────────────────────────────┘
```

**Loss 구성** (Single + 추가):
| Loss | Weight | 설명 |
|------|--------|------|
| loss_backbone_latant | 1.0 | **MSE(feat1, feat2)** - backbone feature 일치 |
| loss_backbone_heatmap | 1.0 | Sub backbone heatmap GT supervision |
| (+ Single의 모든 loss) | | |

---

## 문제 분석

**현상**: Dual (44.91mm) > Single (42.07mm) - Dual이 2.84mm 더 나쁨

### 문제 1: Mutual Learning 방식

1. `loss_backbone_latant = MSE(feat1, feat2)`: 두 backbone feature를 **무조건 동일하게** 강제
2. COCO와 MPII의 서로 다른 feature 분포가 충돌
3. 각 pretrained의 고유한 강점이 상쇄됨
4. epoch 1부터 mutual learning 강제 → pretrained knowledge 손상

### 문제 2: Heatmap의 3D 정보 인코딩 한계 (구조적)

```
Backbone feat [2048, 8, 8]  ← depth/texture/context 정보 풍부
       ↓ (Deconv)
Heatmap [16, 47, 47]        ← 2D 위치만 남음 (depth 손실!)
       ↓ (Encoder)
Z [64]                      ← 극단적 압축
       ↓
3D Pose                     ← depth ambiguity 발생
```

**근거**:
- Heatmap은 본질적으로 **2D 위치의 확률 분포** (x, y만 표현)
- 3D depth 정보가 implicit하게만 존재 → lifting 시 ambiguity
- Backbone feature의 depth cues가 heatmap 단계에서 손실됨

**해결 방향**: Backbone feature를 별도 경로로 보존하여 depth 정보 활용

---

## 실험 단계

### Phase 1: Progressive Warmup (현재 구현 완료)

**목적**: Pretrained knowledge 보존하면서 mutual learning 도입

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| mutual_warmup_epochs | 5 | mutual learning 없음 |
| mutual_rampup_epochs | 10 | 0→1 선형 증가 |
| max_epochs | 20 | 5+10+5 |

**예상 효과**:
- 초기 학습에서 COCO/MPII 각각의 강점 보존
- 갑작스러운 gradient 충돌 방지

**성공 기준**: Dual COCO+MPII (원본) 대비 MPJPE 개선

---

### Phase 2: Ensemble Teacher

**목적**: 두 backbone의 출력을 앙상블하여 pseudo-teacher 생성

**구현 위치**: `custom_egopose_baselinel1_head_multi_backbone_v3.py`

**변경 사항**:
```python
# 현재 (v2)
loss_backbone_latant = MSE(feat1, feat2) * warmup_weight

# 개선 (v3)
conf1 = heatmap1.max().mean()
conf2 = heatmap2.max().mean()
w1 = conf1 / (conf1 + conf2)
w2 = conf2 / (conf1 + conf2)

feat_ensemble = w1 * feat1.detach() + w2 * feat2.detach()
loss_ensemble = MSE(feat1, feat_ensemble) + MSE(feat2, feat_ensemble)
loss_backbone_latant = loss_ensemble * warmup_weight
```

**예상 효과**:
- Confidence 높은 backbone에 더 큰 가중치
- 두 pretrained의 강점 결합

**실험 조합**:
| 실험 | Warmup | Ensemble | 비고 |
|------|--------|----------|------|
| 2-A | ✓ | ✓ | Phase 1 + Ensemble |
| 2-B | ✗ | ✓ | Ensemble만 |

---

### Phase 3: Heatmap KL Divergence

**목적**: Deep Mutual Learning 논문 방식 적용

**구현 위치**: `custom_egopose_baselinel1_head_multi_backbone_v4.py`

**변경 사항**:
```python
def heatmap_kl_divergence_loss(heatmap1, heatmap2, temperature=4.0):
    h1_flat = heatmap1.view(B, K, -1) / temperature
    h2_flat = heatmap2.view(B, K, -1) / temperature

    p1 = F.softmax(h1_flat, dim=-1)
    p2 = F.softmax(h2_flat, dim=-1)

    kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2 * (temperature ** 2)
```

**하이퍼파라미터**:
| 파라미터 | 후보값 | 설명 |
|---------|--------|------|
| temperature | 2, 4, 8 | Soft target 정도 |
| loss_weight | 0.1, 0.5, 1.0 | KL loss 가중치 |

**실험 조합**:
| 실험 | Warmup | Ensemble | KL Div | 비고 |
|------|--------|----------|--------|------|
| 3-A | ✓ | ✗ | ✓ | Warmup + KL |
| 3-B | ✓ | ✓ | ✓ | 전체 조합 |

---

### Phase 4: One-way KD (옵션)

**목적**: Inference에서 main backbone만 사용할 경우

**변경 사항**:
```python
# Sub → Main 방향으로만 KD
loss_kd = MSE(feat_main, feat_sub.detach())
# Sub는 GT에서만 학습 (gradient 차단)
```

**사용 시나리오**:
- 추론 시 sub backbone 제거하여 속도 2배
- Main backbone 성능 최대화

---

### Phase 5: 구조적 개선 (장기)

#### 5-A: HeatmapDecoder 최적화

**문제**: `linear3`가 37.77M 파라미터 (Head 61.4M 중 65%)

**해결**: Conv 기반 decoder로 교체
- 40M → ~1.5M (96% 감소)
- 메모리: 22GB → 16GB

#### 5-B: 2D→3D Lifting (구현 완료 ✅)

**변경**:
```
현재: Heatmap → Encoder → Z[64] → Decoder → 3D
개선: Heatmap → soft_argmax → 2D[16,2] + conf[16] → Lifting → 3D
```

**장점**:
- HeatmapDecoder 완전 제거 (40M → 4M)
- Martinez baseline (검증된 방식)
- 해석 가능한 중간 표현
- Heatmap MSE + Coord MSE 동시 학습

**구현 파일**:
- Head: `custom_egopose_lifting_head.py`
- Config: `HMD_xregopose_single_lifting_config.py`

**비교 대상 (중요!)**:

| 항목 | Single COCO (기존 베스트) | Single Lifting (신규) |
|------|--------------------------|----------------------|
| Config | `HMD_xregopose_single_coco_full_config.py` | `HMD_xregopose_single_lifting_config.py` |
| Head | `CustomxRegoposeBaselinel1` | `CustomEgoposeLiftingHead` |
| Backbone | ResNet-101 COCO | ResNet-101 COCO (동일) |
| 2D→3D | Encoder→Z[64]→Decoder | soft_argmax→Lifting |
| Head Params | ~61M | ~13M |
| MPJPE | **42.07mm** | ? (실험 필요) |

> **주의**: Lifting은 Dual이 아닌 **Single backbone**과 비교해야 함!
> Dual (44.91mm)과 비교하면 구조 차이가 너무 커서 의미 없음.

---

### Phase 6: Backbone Feature Fusion (핵심 구조 개선)

**문제 인식**: Heatmap은 2D 위치 정보에 최적화되어 있어 3D depth 정보 인코딩이 어려움

**학술적 근거**:
- [Depth Ambiguity Survey](https://www.mdpi.com/2076-3417/12/20/10591): "하나의 2D pose → 여러 3D pose 매핑 가능"
- [EgoTAP](https://arxiv.org/html/2402.18330): "CNN encoder가 heatmap 정보 제대로 보존 못함"
- [Lifting by Image](https://arxiv.org/abs/2312.15636): "image의 semantic/texture 정보가 lifting에 기여"

#### 6-A: Backbone + Heatmap Latent Concat (추천 - 먼저 시도)

**구조**:
```
Backbone feat [2048, 8, 8]
       │
       ├──→ GAP → FC → Z_backbone [256]  ← depth/context cues
       │                      │
       ↓ (Deconv)             │
Heatmap [16, 47, 47]          │
       ↓ (Encoder)            │
Z_heatmap [64] ← 2D 위치      │
       │                      │
       └──── Concat ──────────┘
              ↓
         [64 + 256 + 64(HMD)] = [384]
              ↓
         Pose Decoder → 3D Pose
```

**역할 분리**:
| Component | 역할 |
|-----------|------|
| Z_heatmap | 정확한 2D 관절 위치 (x, y) |
| Z_backbone | 깊이/텍스처/컨텍스트 (depth cues) |
| Z_hmd | 머리/손 3D 위치 (absolute reference) |

**구현 위치**: `custom_egopose_baselinel1_head_multi_backbone_v5.py` (또는 v3에 통합)

**Config**: `HMD_xregopose_h5cache_coco_mpii_backbone_fusion_config.py`

#### 6-B: 2D Coords + Backbone Feature

**구조**:
```
Heatmap → soft-argmax → 2D coords [16, 2] + conf [16]
                              │
Backbone feat → GAP → FC → Context [256]
                              │
              Concat ─────────┘
                 ↓
           [32 + 16 + 256 + 9(HMD)] = [313]
                 ↓
           Lifting Network → 3D Pose
```

**장점**: 2D 좌표가 명시적, backbone이 depth 해결에 직접 기여

---

### 실험 조합 요약

| Phase | 개선 방향 | 구현 상태 | 비교 대상 |
|-------|----------|----------|----------|
| 1 | Progressive Warmup | ✅ 완료 | vs Dual COCO+MPII (44.91mm) |
| 2 | Ensemble Teacher | ❌ 미구현 | vs Dual |
| 3 | KL Divergence | ❌ 미구현 | vs Dual |
| 4 | One-way KD | ❌ 미구현 | vs Dual |
| **5-A** | HeatmapDecoder 최적화 | ❌ 미구현 | - |
| **5-B** | **2D→3D Lifting** | **✅ 완료 (190mm - 실패)** | **vs Single COCO (42.07mm)** |
| **5-C** | **Lifting + Backbone Fusion** | **❌ 미구현** | **vs 5-B** |
| **5-D** | **Attention Lifting** | **❌ 미구현** | **vs 5-C** |
| 6 | Backbone Feature Fusion (기존 구조) | ❌ 미구현 | vs Single/Dual |

---

### Phase 5-C: Lifting + Backbone Feature Fusion (계획)

**목적**: Lifting Head에 Backbone feature를 추가하여 depth cues 활용

**구조**:
```
Backbone feat [2048, 8, 8]
       │
       ├───────────────────────────┐
       ↓ (Deconv)                  ↓ (GAP → FC)
Heatmap [16, 47, 47]          Z_backbone [256]
       ↓ (soft_argmax)             │
2D [32] + conf [16]                │ (depth cues!)
       │                           │
       └────── Concat ─────────────┘
                  ↓
       [32 + 16 + 256 + 9] = 313
                  ↓
         Lifting Network → 3D
```

**핵심 가설**:
- Backbone feature는 texture/context에서 depth 정보를 implicit하게 인코딩
- Heatmap (2D 위치) + Backbone (depth) → 상호 보완

**구현 파일** (예정):
- Head: `custom_egopose_lifting_backbone_fusion_head.py`
- Config: `HMD_xregopose_single_lifting_backbone_config.py`

**비교 대상**: Phase 5-B Lifting 결과 (not Single COCO directly)

---

### Phase 5-D: Attention 기반 Lifting (계획)

**목적**: Cross-Attention으로 각 관절이 backbone에서 필요한 depth 정보를 선택적으로 쿼리

**구조**:
```
2D coords [16, 2] → Joint queries [16, D]
                          ↓
Backbone [2048,8,8] → Spatial tokens [64, D]
                          ↓
              Cross-Attention (Q: joints, K/V: backbone)
                          ↓
              Depth-aware joints [16, D]
                          ↓
                     + HMD → 3D Pose
```

**핵심 설계**:
```python
# Gradient 흐름 설계
coords_2d = soft_argmax(heatmaps).detach()  # 2D: gradient 차단
joint_q = embed(coords_2d)                   # Query

backbone_kv = backbone_proj(backbone_feat)   # K/V: gradient 흐름!

depth_joints = cross_attention(joint_q, backbone_kv, backbone_kv)
pose_3d = output(depth_joints)
```

**장점**:
1. 각 관절이 **필요한 spatial 위치에서 depth 쿼리**
2. `attn_weights`로 **어느 위치 참조했는지 해석 가능**
3. **역할 분리**: 2D coords=위치(Query), Backbone=depth(Key/Value)

**구현 파일** (예정):
- Head: `custom_egopose_attention_lifting_head.py`
- Config: `HMD_xregopose_attention_lifting_config.py`

**비교 대상**: Phase 5-C Backbone Fusion 결과

**추천 실험 순서**:
1. Phase 1 (Warmup) - 이미 구현, 실험 진행
2. **Phase 6-A (Backbone Fusion)** - 핵심 구조 개선, 병렬 진행 가능
3. Phase 1 + 6-A 조합
4. 결과에 따라 Phase 2, 3 선택적 적용

---

## 실험 순서 및 일정

| 순서 | 실험 | 예상 소요 | 의존성 |
|------|------|----------|--------|
| 1 | Phase 1: Progressive Warmup | 1일 | 없음 (구현 완료) |
| 2 | Baseline: Dual COCO+MPII (원본) | 0.5일 | Phase 1과 병렬 |
| 3 | Phase 2-A: Warmup + Ensemble | 1일 | Phase 1 완료 후 |
| 4 | Phase 3-A: Warmup + KL Div | 1일 | Phase 1 완료 후 |
| 5 | Phase 3-B: 전체 조합 | 1일 | Phase 2, 3 결과 확인 후 |
| 6 | Phase 5-A: Decoder 최적화 | 2일 | 최적 조합 확정 후 |

---

## 평가 메트릭

### 주요 메트릭
- **Full Body MPJPE** (mm): 주 평가 지표
- **Upper Body MPJPE** (mm)
- **Lower Body MPJPE** (mm)

### 보조 메트릭
- `mutual_weight`: Progressive warmup 진행 상태
- `loss_backbone_latant`: Mutual learning loss 값
- `acc_pose`: 2D heatmap 정확도

### 모니터링 (Wandb)
- Loss curves: 각 loss 컴포넌트별 추이
- Learning rate schedule
- GPU 메모리 사용량

---

## Config 파일 명명 규칙

```
HMD_xregopose_h5cache_coco_mpii_{variant}_config.py
```

| Variant | 설명 |
|---------|------|
| (없음) | 원본 dual backbone |
| warmup | Phase 1: Progressive warmup |
| ensemble | Phase 2: Ensemble teacher |
| kldiv | Phase 3: KL divergence |
| full | 전체 조합 |
| lifting | Phase 5-B: 2D→3D lifting |

---

## 결과 기록 템플릿

### 실험: Phase X - [실험명]

**Config**: `HMD_xregopose_h5cache_coco_mpii_xxx_config.py`

**하이퍼파라미터**:
| 파라미터 | 값 |
|---------|-----|
| mutual_warmup_epochs | |
| mutual_rampup_epochs | |
| temperature | |
| ... | |

**결과**:
| Metric | Value |
|--------|-------|
| Full Body MPJPE | mm |
| Upper Body MPJPE | mm |
| Lower Body MPJPE | mm |
| Best Epoch | |

**관찰**:
-

**결론**:
-

---

## 참고 논문

1. **Deep Mutual Learning** (Zhang et al., 2017) - arXiv:1706.00384
2. **Knowledge Distillation** (Hinton et al., 2015) - arXiv:1503.02531
3. **A Simple Baseline for 3D Pose** (Martinez et al., 2017) - ICCV 2017

---

## 현재 진행 상황

- [x] Phase 1: Progressive Warmup 구현 (Dual backbone용)
  - [x] Head v2 생성: `custom_egopose_baselinel1_head_multi_backbone_v2.py`
  - [x] MutualLearningWarmupHook 생성
  - [x] Config 생성: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`
- [x] **Phase 5-B: 2D→3D Lifting 구현 (Single backbone용)**
  - [x] **Head 생성: `custom_egopose_lifting_head.py`**
  - [x] **Config 생성: `HMD_xregopose_single_lifting_config.py`**
  - [x] **실험 결과: 190mm (실패 - depth 정보 부족)**
- [ ] Phase 1 실험 실행 (비교: Dual 44.91mm)
- [ ] **Phase 5-C: Lifting + Backbone Fusion 구현**
  - [ ] Head 생성: `custom_egopose_lifting_backbone_fusion_head.py`
  - [ ] Config 생성: `HMD_xregopose_single_lifting_backbone_config.py`
- [ ] **Phase 5-D: Attention Lifting 구현**
  - [ ] Head 생성: `custom_egopose_attention_lifting_head.py`
  - [ ] Config 생성: `HMD_xregopose_attention_lifting_config.py`
- [ ] Phase 2, 3 구현 (필요시)
