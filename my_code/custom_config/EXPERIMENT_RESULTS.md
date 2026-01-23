# EgoPose 3D 실험 결과 및 분석

> 최종 업데이트: 2026-01-23

## 목표

**Single COCO Baseline (41.37mm MPJPE)보다 나은 3D pose estimation 성능 달성**

---

## 실험 결과 요약

### 전체 비교표

| # | 실험명 | Config | Head | MPJPE (mm) | Best Epoch | 상태 |
|---|--------|--------|------|------------|------------|------|
| 0 | **Single COCO (Baseline)** | `HMD_xregopose_single_coco_full_config.py` | `CustomxRegoposeBaselinel1` | **41.37** | 8 | 🏆 Best |
| 1 | Dual COCO+MPII | `HMD_xregopose_h5cache_coco_mpii_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | 43.26 | 8 | ❌ |
| 2 | Dual Warmup v2 | `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | `CustomxRegoposeBaselinel1_multi_backbone_v2` | 45.93 | 9 | ❌ |
| 3 | Single Lifting | `HMD_xregopose_single_lifting_config.py` | `CustomEgoposeLiftingHead` | 45.92 | 9 | ❌ |
| 4 | Lifting + Backbone Fusion | `HMD_xregopose_lifting_backbone_fusion_config.py` | `CustomEgoposeLiftingBackboneFusionHead` | 105.18 | 5 | ❌ (7ep 중단) |
| 5 | EfficientHeatmapDecoder | `HMD_xregopose_efficient_decoder_full_config.py` | `CustomxRegoposeBaselinel1` | 45.06 | 8 | ❌ (param 효율화) |
| 6 | Attention Lifting v1 | `HMD_xregopose_attention_lifting_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 45.43 | 7 | ❌ (epoch 4 불안정) |
| 7 | Attention Lifting v2 | `HMD_xregopose_attention_lifting_v2_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 48.38 | 8 | ❌ (v1보다 악화) |
| 8 | Attention Lifting v3 | `HMD_xregopose_attention_lifting_v3_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 46.68 | 1 | ❌ (epoch 2 발산) |
| 9 | Attention Lifting v4 | `HMD_xregopose_attention_lifting_v4_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 47.95 | 6 | ❌ (v1보다 나쁨) |

### 부위별 결과 상세

| 실험명 | Full Body | Upper Body | Lower Body | Best Epoch |
|--------|-----------|------------|------------|------------|
| **Single COCO (Baseline)** | **41.37mm** | **29.42mm** | **53.31mm** | 8 |
| Dual COCO+MPII | 43.26mm | 30.03mm | 56.48mm | 8 |
| Dual Warmup v2 | 45.93mm | 31.07mm | 60.79mm | 9 |
| Single Lifting | 45.92mm | 33.91mm | 57.93mm | 9 |
| EfficientHeatmapDecoder | 45.06mm | 30.54mm | 59.58mm | 8 |
| Attention Lifting v1 | 45.43mm | 30.14mm | 60.72mm | 7 |
| Attention Lifting v2 | 48.38mm | 32.03mm | 64.73mm | 8 |
| Attention Lifting v3 | 46.68mm | 34.41mm | 58.95mm | 1 |
| Attention Lifting v4 | 47.95mm | 30.10mm | 65.80mm | 6 |

### Baseline 대비 비교

| 실험명 | Full Body | vs Baseline | 비고 |
|--------|-----------|-------------|------|
| **Single COCO (Baseline)** | **41.37mm** | - | 🏆 |
| Dual COCO+MPII | 43.26mm | +1.89mm ❌ | mutual learning 악화 |
| Dual Warmup v2 | 45.93mm | +4.56mm ❌ | warmup도 효과 없음 |
| Single Lifting | 45.92mm | +4.55mm ❌ | depth 정보 부족 |
| EfficientHeatmapDecoder | 45.06mm | +3.69mm ❌ | 96% param 절감, 성능 하락 |
| Attention Lifting v1 | 45.43mm | +4.06mm ❌ | epoch 4 spike, 훈련 불안정 |
| Attention Lifting v2 | 48.38mm | +7.01mm ❌ | warmup 역효과, v1보다 악화 |
| Attention Lifting v3 | 46.68mm | +5.31mm ❌ | LR 과다, epoch 2 발산 |
| Attention Lifting v4 | 47.95mm | +6.58mm ❌ | CosineAnnealing, v1보다 나쁨 |

---

## 실험 상세 결과

### 실험 0: Single COCO Baseline 🏆

**Config**: `HMD_xregopose_single_coco_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_single_coco_full`

**구조**:
```
Backbone: ResNet-101 (COCO pretrained)
    ↓
Head: CustomxRegoposeBaselinel1
    ├── Deconv → Heatmap [16, 47, 47]
    ├── HeatmapEncoder → Z [64]
    ├── + HMD info [9→64]
    ├── PoseDecoder → 3D Pose [16, 3]
    └── HeatmapDecoder → Recon Heatmap (40M params)
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 6 | 46.92mm | 33.15mm | 60.69mm |
| 7 | 43.69mm | 30.95mm | 56.43mm |
| **8** | **41.37mm** | **29.42mm** | **53.31mm** |
| 9 | 42.07mm | 29.15mm | 55.00mm |
| 10 | 41.39mm | 29.04mm | 53.74mm |

**핵심 특징**:
- Single backbone (COCO pretrained only)
- HeatmapEncoder→Z[64]→Decoder 구조
- Head 파라미터: ~61M (HeatmapDecoder 40M 포함)

---

### 실험 1: Dual COCO+MPII

**Config**: `HMD_xregopose_h5cache_coco_mpii_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_coco_mpii`

**가설**: 두 개의 pretrained backbone을 mutual learning으로 결합하면 성능 향상

**구조**:
```
Backbone1: ResNet-101 (COCO)  |  Backbone2: ResNet-101 (MPII)
         ↓                    |           ↓
    feat1 [2048,8,8]          |      feat2 [2048,8,8]
         ↓                    |           ↓
    Heatmap1                  |      Heatmap2
         ↓                    |           ↓
         └──── MSE Loss ──────┘  ← loss_backbone_latant
         ↓
    Main path → 3D Pose
```

**Epoch별 결과** (10 epoch run):
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 4 | 45.42mm | 31.39mm | 59.45mm |
| 6 | 44.25mm | 30.48mm | 58.02mm |
| **8** | **43.26mm** | **30.03mm** | **56.48mm** |
| 10 | 44.91mm | 30.56mm | 59.26mm |

**분석**:
- Dual backbone이 Single보다 **오히려 나쁨** (+1.89mm)
- `MSE(feat1, feat2)` mutual learning이 문제:
  - COCO/MPII의 서로 다른 feature 분포가 충돌
  - 각 pretrained의 고유한 강점이 상쇄됨

---

### 실험 2: Dual Warmup v2 (Progressive Warmup)

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_coco_mpii_warmup_10ep`

**가설**: 초기 epoch에서 mutual learning을 점진적으로 도입하면 pretrained knowledge 보존

**Warmup 스케줄**:
```
Epoch 0-1:  mutual_weight = 0.0  (warmup)
Epoch 2-6:  mutual_weight = 0.0 → 0.8  (ramp-up)
Epoch 7-9:  mutual_weight = 1.0  (full)
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | mutual_weight |
|-------|-----------|------------|------------|---------------|
| 6 | 49.56mm | 33.27mm | 65.85mm | 0.8 |
| 7 | 50.40mm | 33.88mm | 66.91mm | 1.0 |
| 8 | 46.78mm | 32.04mm | 61.53mm | 1.0 |
| **9** | **45.93mm** | **31.07mm** | **60.79mm** | 1.0 |
| 10 | 49.47mm | 32.39mm | 66.55mm | 1.0 |

**결론**:
- Progressive warmup만으로는 Dual의 근본적 문제 해결 불가
- 기존 Dual(43.26mm)보다도 **나쁨** (+2.67mm)
- Mutual learning 자체가 COCO/MPII 조합에서 비효율적

---

### 실험 3: Single Lifting (Soft-argmax 2D→3D)

**Config**: `HMD_xregopose_single_lifting_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_single_lifting`

**가설**: HeatmapEncoder/Decoder 대신 soft-argmax + Lifting으로 구조 단순화

**구조**:
```
[기존 Single COCO]
Backbone → Heatmap → Encoder → Z[64] → PoseDecoder → 3D
                            ↘ HeatmapDecoder (40M params!)

[Single Lifting]
Backbone → Heatmap → soft_argmax → 2D[32] + conf[16]
                                        ↓
                         Lifting Network (4M params)
                                        ↓
                                   3D Pose
```

**Epoch별 결과** (학습 불안정!):
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 149.54mm | 89.33mm | 209.75mm | 초기 |
| 2 | 84.47mm | 58.08mm | 110.87mm | 개선 |
| **3** | **190.54mm** | 146.27mm | 234.81mm | **Spike!** |
| 4 | 80.44mm | 51.69mm | 109.19mm | 회복 |
| 5 | 57.09mm | 37.58mm | 76.60mm | |
| **9** | **45.92mm** | **33.91mm** | **57.93mm** | Best |
| 10 | 53.69mm | 41.58mm | 65.80mm | Overfitting |

**실패 원인 분석**:
1. **Depth 정보 부재**: 2D 좌표만으로는 3D depth 예측 불가 (depth ambiguity)
2. **학습 불안정**: Epoch 3에서 190mm spike 발생
3. **Backbone feature 미활용**: depth/texture/context 정보가 3D lifting에 전달 안됨

**핵심 인사이트**:
> "Heatmap에 3D 정보가 섞이는 것 같음. 이를 분리시키는 게 더 좋지 않겠나?"
> - Heatmap → 2D 위치 전담 (gradient 차단)
> - Backbone → 3D depth cues 전담 (gradient 흐름)

---

### 실험 4: Lifting + Backbone Fusion (준비 완료)

**Config**: `HMD_xregopose_lifting_backbone_fusion_config.py`

**가설**: Backbone feature를 별도 경로로 보존하여 depth 정보 활용

**구조**:
```
Backbone feat [2048, 8, 8]
       │
       ├─────────────────────────────┐
       │                             │
       ↓ (Deconv)                    ↓ (GAP → FC)
Heatmap [16, 47, 47]            Z_backbone [256]
       │                             │
       │  ← 2D 위치 (gradient 차단)   │  ← 3D depth (gradient 흐름)
       │                             │
       ↓ (soft_argmax)               │
2D coords [32] + conf [16]           │
       │                             │
       └───────── Concat ────────────┘
                    ↓
           [32 + 16 + 256 + 9] = 313
                    ↓
             Lifting Network
                    ↓
                3D Pose
```

**핵심 설계**:
```python
# 역할 분리 (Role Separation)
coords_2d = soft_argmax(heatmaps)
coords_2d_detached = coords_2d.detach()  # 2D: gradient 차단

z_backbone = backbone_encoder(backbone_feat)  # depth: gradient 흐름!

pose_3d = lifting_network(coords_2d_detached, confidence, z_backbone, hmd_info)
```

**예상 효과**:
- Heatmap: 2D 위치만 학습 (heatmap MSE + coord MSE)
- Backbone: 3D depth cues 학습 (3D loss만)
- 역할 분리로 학습 안정화 기대

**상태**: ❌ 실패 (105.18mm, 7ep 중단)

---

### 실험 5: EfficientHeatmapDecoder (파라미터 효율화)

**Config**: `HMD_xregopose_efficient_decoder_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_efficient_decoder_full`

**가설**: FC-heavy HeatmapDecoder를 Conv-based로 교체하면 파라미터 효율화 (40M → 1.35M)

**구조 비교**:
```
[Original HeatmapDecoder - 40M params]
Z[64] → FC(64→47*47*16) → reshape → Heatmap[16,47,47]
         ↑ 약 2.3M params (출력만)

[EfficientHeatmapDecoder - 1.35M params]
Z[64] → FC(64→256*6*6) → reshape → [256,6,6]
                                       ↓
                           ConvTranspose2d (256→128, 3×3, s2)
                                       ↓ [128,12,12]
                           ConvTranspose2d (128→64, 3×3, s2)
                                       ↓ [64,24,24]
                           ConvTranspose2d (64→16, 3×3, s2)
                                       ↓ [16,48,48]
                           AdaptiveAvgPool2d → [16,47,47]
```

**파라미터 비교**:
| 항목 | Original | Efficient | 절감율 |
|------|----------|-----------|--------|
| HeatmapDecoder | 40.0M | 1.35M | **96.6%** |
| 전체 Head | ~61M | ~22M | ~64% |

**Epoch별 결과** (10 epoch):
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 82.52mm | 59.72mm | 105.32mm |
| 2 | 58.12mm | 38.93mm | 77.31mm |
| 3 | 52.10mm | 34.12mm | 70.08mm |
| 4 | 49.93mm | 33.69mm | 66.17mm |
| 5 | 47.93mm | 32.00mm | 63.86mm |
| 6 | 48.82mm | 33.17mm | 64.48mm |
| 7 | 46.90mm | 31.83mm | 61.97mm |
| **8** | **45.06mm** | **30.54mm** | **59.58mm** |
| 9 | 46.01mm | 31.50mm | 60.52mm |
| 10 | 46.92mm | 31.77mm | 62.08mm |

**분석**:
- **파라미터 효율화 성공**: 96.6% 절감 (40M → 1.35M)
- **성능 하락**: +3.69mm vs Baseline (41.37mm → 45.06mm)
- **Trade-off**: 파라미터 30배 절감 vs 성능 9% 하락

**실패 원인 분석**:
1. **표현력 부족**: ConvTranspose2d의 점진적 업샘플링이 Z→Heatmap 직접 매핑보다 표현력 제한
2. **Checkerboard Artifact**: ConvTranspose2d 고유의 artifact가 heatmap 품질 저하
3. **정보 손실**: 6×6 → 47×47 업샘플링 과정에서 정보 손실

**개선 아이디어**:
- AdaIN (Adaptive Instance Normalization): Z가 각 레이어에 영향
- PixelShuffle: Checkerboard artifact 방지
- Joint-wise Generation: 관절별 독립 decoder

**결론**:
> EfficientHeatmapDecoder는 파라미터 효율화에는 성공했으나,
> 성능 하락이 발생. 순수 효율화보다는 구조적 개선이 필요.

---

### 실험 6: Attention Lifting v1 (Cross-Attention 기반)

**Config**: `HMD_xregopose_attention_lifting_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_full`
**Note**: Attention Lifting v1~v4는 EfficientHeatmapDecoder 사용

**가설**: Cross-Attention으로 backbone에서 depth 정보를 query하고, HMD 정보와 joint 간 관계를 attention으로 학습

**구조**:
```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓ (Deconv)                         ↓ (reshape → spatial tokens)
Heatmap [16, 47, 47]               Backbone Spatial [64, 256]
       │                                  │
       ↓ (soft_argmax)                    │
2D coords [16, 2]                         │
       │                                  │
       ↓ (Joint Embedding)                │
Joint tokens [16, 64]                     │
       │                                  │
       ↓ (Backbone Cross-Attention) ←─────┘  Query depth from backbone
       ↓
       ↓ (HMD Cross-Attention) ←───── HMD tokens [3, 64] (head, R hand, L hand)
       ↓
       ↓ (Joint Self-Attention × 2) ←── 관절 간 구조적 관계 학습
       ↓
3D Pose [16, 3]
```

**핵심 설계**:
```python
# Attention 기반 Lifting
# 1. Backbone Cross-Attention: joint가 backbone spatial feature에서 depth query
# 2. HMD Cross-Attention: joint가 HMD 3개 토큰(head, R hand, L hand)에서 reference 학습
# 3. Joint Self-Attention: 관절 간 구조적 관계 (대칭, 연결) 학습

# EfficientHeatmapDecoder로 Z 정규화 (reconstruction loss)
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | LR | 비고 |
|-------|-----------|------------|------------|-----|------|
| 1 | 55.30mm | 42.63mm | 67.97mm | 5e-4 | |
| 2 | 53.81mm | 37.84mm | 69.79mm | 5e-4 | |
| 3 | 49.19mm | 33.74mm | 64.63mm | 5e-4 | |
| **4** | **52.12mm** | 34.79mm | 69.44mm | 5e-4 | **⚠️ Spike!** |
| 5 | 47.88mm | 32.49mm | 63.28mm | 2.5e-4 | LR decay |
| 6 | 47.86mm | 31.87mm | 63.84mm | 2.5e-4 | |
| **7** | **45.43mm** | **30.14mm** | **60.72mm** | 2.5e-4 | **Best** |
| 8 | 45.74mm | 30.35mm | 61.14mm | 1.25e-4 | LR decay |
| 9 | 45.60mm | 29.97mm | 61.22mm | 1.25e-4 | |
| 10 | 45.63mm | 29.66mm | 61.61mm | 1.25e-4 | |

**훈련 불안정 분석**:
- **Epoch 4 Spike**: 49.19mm → 52.12mm (+2.93mm 급등)
  - LR milestone=[4,7]로 epoch 4에서 LR 변화 없이 불안정
  - Attention weight가 급격히 변화하면서 발생
- **Epoch 5 회복**: LR=2.5e-4로 감소 후 안정화 (52.12 → 47.88mm)

**LR Schedule 분석**:
```
Epoch 1-4: LR = 5e-4 (높은 LR에서 불안정)
Epoch 5-7: LR = 2.5e-4 (LR decay 후 안정화, Best 달성)
Epoch 8-10: LR = 1.25e-4 (수렴)
```

**실패 원인 분석**:
1. **Attention 학습 불안정**: Cross-attention이 초기에 급격한 weight 변화
2. **LR Schedule 부적절**: MultiStepLR [4,7]이 attention에 맞지 않음
3. **Warmup 부재**: 초기 LR이 높아 attention weight 불안정

**개선 방향 (v2)**:
1. **LR Warmup**: LinearLR (0.1x → 1x, 2 epochs)
2. **CosineAnnealing**: MultiStepLR 대신 부드러운 LR decay
3. **Gradient Clipping**: max_norm=1.0으로 gradient 폭발 방지

**v2 Config**: `HMD_xregopose_attention_lifting_v2_full_config.py`
```python
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=2, end=10, eta_min=1e-5),
]
```

**결론**:
> Attention Lifting은 구조적으로 합리적이나, 훈련 안정성 문제가 있음.
> v2에서 LR warmup + CosineAnnealing + Gradient Clipping으로 안정화 예정.

---

### 실험 7: Attention Lifting v2 (Warmup + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v2_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v2_full`

**가설**: LR warmup + CosineAnnealing + Gradient Clipping으로 v1의 훈련 불안정 해결

**v1 대비 변경사항**:
```python
# v1: MultiStepLR [4, 7], no warmup, no gradient clipping
# v2:
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # 추가
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=2),  # Warmup 추가
    dict(type='CosineAnnealingLR', begin=2, end=10, eta_min=1e-5),  # 변경
]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | LR | 비고 |
|-------|-----------|------------|------------|-----|------|
| 1 | 111.34mm | 87.46mm | 135.22mm | 5e-5 | Warmup (0.1x) |
| 2 | 59.15mm | 40.85mm | 77.45mm | 5e-4 | Warmup 완료 |
| 3 | 56.16mm | 37.94mm | 74.38mm | ~4.5e-4 | CosineAnnealing |
| 4 | 51.80mm | 33.17mm | 70.42mm | ~3.9e-4 | |
| **5** | **57.78mm** | 36.09mm | 79.48mm | ~3.2e-4 | **⚠️ Spike!** |
| 6 | 51.31mm | 33.76mm | 68.86mm | ~2.5e-4 | 회복 |
| 7 | 50.83mm | 33.41mm | 68.25mm | ~1.8e-4 | |
| **8** | **48.38mm** | **32.03mm** | **64.73mm** | ~1.2e-4 | **Best** |
| 9 | 50.92mm | 32.84mm | 69.01mm | ~0.6e-4 | |
| 10 | 48.97mm | 32.49mm | 65.45mm | ~1e-5 | |

**v1 vs v2 비교**:
| 항목 | v1 (MultiStepLR) | v2 (Warmup+Cosine) | 비고 |
|------|------------------|-------------------|------|
| Best MPJPE | **45.43mm** | 48.38mm | v1이 2.95mm 더 좋음 |
| Best Epoch | 7 | 8 | |
| Epoch 1 | 55.30mm | 111.34mm | **Warmup이 초기 학습 지연** |
| Spike | Epoch 4 (+2.93mm) | Epoch 5 (+5.98mm) | v2 spike가 더 심함 |

**실패 원인 분석**:
1. **Warmup 역효과**: LR=0.1x로 시작하여 epoch 1 에러가 111mm (v1: 55mm)
   - 2 epoch 동안 충분히 학습하지 못함
   - v1은 처음부터 높은 LR로 빠르게 감소
2. **CosineAnnealing 부적합**: 부드러운 LR decay가 attention에 맞지 않음
   - MultiStepLR의 급격한 LR drop이 오히려 효과적
3. **여전히 불안정**: Epoch 5에서 spike 발생 (57.78mm, +5.98mm)
   - Gradient clipping이 spike를 완전히 방지하지 못함

**핵심 인사이트**:
> - **Warmup은 역효과**: Attention Lifting은 초기 빠른 학습이 중요
> - **MultiStepLR이 더 적합**: CosineAnnealing보다 급격한 LR drop이 효과적
> - **다음 시도**: 더 높은 LR (0.001)로 v1 스케줄 유지 (v3)

**v3 계획** (LR=0.001):
```python
# v3: Higher LR (2x of v1) + MultiStepLR + Gradient Clipping
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[4, 7], gamma=0.5, by_epoch=True),
]
```

---

### 실험 8: Attention Lifting v3 (Higher LR = 0.001)

**Config**: `HMD_xregopose_attention_lifting_v3_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v3_full`

**가설**: 높은 LR (0.001 = v1의 2배)로 빠른 수렴 + MultiStepLR 유지

**v1 대비 변경사항**:
```python
# v1: LR=0.0005, no gradient clipping
# v3:
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),  # 2x LR
    clip_grad=dict(max_norm=1.0, norm_type=2),  # 추가
)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[4, 7], gamma=0.5, by_epoch=True),  # 동일
]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | LR | 비고 |
|-------|-----------|------------|------------|-----|------|
| **1** | **46.68mm** | **34.41mm** | **58.95mm** | 1e-3 | **Best** |
| 2 | 61.09mm | 37.27mm | 84.92mm | 1e-3 | **⚠️ +14.4mm 발산!** |
| 3 | 58.99mm | 34.35mm | 83.64mm | 1e-3 | |
| 4 | 54.63mm | 31.45mm | 77.80mm | 1e-3 | |
| 5 | 62.43mm | 36.71mm | 88.15mm | 5e-4 | **⚠️ LR decay 후 재발산!** |
| 6 | 51.61mm | 32.43mm | 70.79mm | 5e-4 | |
| 7 | 52.39mm | 31.70mm | 73.08mm | 5e-4 | |
| 8 | 51.02mm | 31.33mm | 70.71mm | 2.5e-4 | |
| 9 | 51.11mm | 31.17mm | 71.05mm | 2.5e-4 | |
| 10 | 52.95mm | 33.18mm | 72.72mm | 2.5e-4 | |

**v1 vs v2 vs v3 비교**:
| 항목 | v1 (LR=5e-4) | v2 (Warmup+Cosine) | v3 (LR=1e-3) |
|------|--------------|-------------------|--------------|
| **Best MPJPE** | **45.43mm 🏆** | 48.38mm | 46.68mm |
| Best Epoch | 7 | 8 | 1 |
| Epoch 1 | 55.30mm | 111.34mm | 46.68mm |
| 최악 Spike | +2.93mm | +5.98mm | **+14.41mm** |
| 수렴 여부 | ✅ 안정적 | ⚠️ 불안정 | ❌ 발산 |

**실패 원인 분석**:
1. **LR 과다**: 0.001이 Attention에 너무 공격적
   - Epoch 1에서 좋은 초기값(46.68mm)을 찾았으나
   - Epoch 2에서 +14.4mm 급등하며 발산
2. **LR decay 후 재발산**: Epoch 5에서 LR 0.5x 감소 직후 또 spike (62.43mm)
3. **회복 불가**: Epoch 1 수준으로 다시 돌아가지 못함

**Attention Lifting 시리즈 최종 결론**:
| 버전 | 설정 | Best MPJPE | 결과 |
|------|------|------------|------|
| **v1** | LR=5e-4, MultiStepLR | **45.43mm** | 🏆 Best |
| v2 | LR=5e-4, Warmup+Cosine | 48.38mm | ❌ Warmup 역효과 |
| v3 | LR=1e-3, MultiStepLR | 46.68mm | ❌ LR 과다, 발산 |
| v4 | LR=1e-3, CosineAnnealing | 47.95mm | ❌ 초기 수렴 방해 |

> **결론**: Attention Lifting v1 (45.43mm)이 최선이나, Baseline (41.37mm)보다 +4.06mm 나쁨.
> Attention 구조 자체의 한계 또는 완전히 다른 접근법 필요.

---

### 실험 9: Attention Lifting v4 (High LR + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v4_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v4_full`

**가설**: v3의 높은 LR (0.001) + v2의 CosineAnnealing (warmup 없이) = 빠른 초기 수렴 + 안정적 감소

**설정**:
```python
# v3의 높은 LR + CosineAnnealing (warmup 없이)
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', begin=0, end=10, eta_min=1e-5, by_epoch=True),
]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 76.12mm | 50.47mm | 101.77mm | |
| 2 | 59.32mm | 36.80mm | 81.84mm | |
| 3 | 54.48mm | 33.33mm | 75.63mm | |
| 4 | 50.70mm | 31.45mm | 69.94mm | |
| 5 | 56.30mm | 35.52mm | 77.09mm | ⚠️ Spike |
| **6** | **47.95mm** | **30.10mm** | **65.80mm** | **Best** |
| 7 | 51.24mm | 32.21mm | 70.27mm | |
| 8 | 49.27mm | 30.37mm | 68.17mm | |
| 9 | 52.01mm | 32.16mm | 71.86mm | |
| 10 | 52.10mm | 32.67mm | 71.52mm | |

**v3 vs v4 비교**:
| 항목 | v3 (MultiStepLR) | v4 (CosineAnnealing) |
|------|------------------|---------------------|
| Epoch 1 | 46.68mm | 76.12mm |
| Best | 46.68mm (ep1) | 47.95mm (ep6) |
| 안정성 | ❌ 발산 | ✅ 안정적 |
| Spike | +14.4mm | +5.6mm |

**실패 원인 분석**:
1. **CosineAnnealing이 초기 수렴 방해**:
   - v3는 LR=0.001을 epoch 1 동안 유지 → 빠른 초기 수렴 (46.68mm)
   - v4는 CosineAnnealing으로 epoch 1 중에도 LR 감소 → 느린 초기 수렴 (76.12mm)
2. **v4는 안정적이나 느림**: Spike가 작고 (+5.6mm) 안정적이지만, v1보다 2.5mm 나쁨

**핵심 인사이트**:
> v3의 epoch 1 성공은 **높은 LR 유지**가 핵심이었음.
> CosineAnnealing은 안정성은 높이나 초기 수렴을 방해함.

---

## 종합 분석

### 실패한 접근법

| 접근법 | 결과 | 문제점 |
|--------|------|--------|
| Dual Backbone + Mutual Learning | 43.26mm (+1.89mm) | COCO/MPII feature 분포 충돌 |
| Progressive Warmup | 45.93mm (+4.56mm) | 근본적 mutual learning 문제 해결 불가 |
| Pure 2D→3D Lifting | 45.92mm (+4.55mm) | Depth 정보 부재, 학습 불안정 |
| Lifting + Backbone Fusion | 105.18mm (+63.81mm) | 역할 분리 실패, gradient 차단 부작용 |
| EfficientHeatmapDecoder | 45.06mm (+3.69mm) | Conv 업샘플링 표현력 부족 |
| Attention Lifting v1 | 45.43mm (+4.06mm) | Epoch 4 spike, LR schedule 부적절 |
| Attention Lifting v2 | 48.38mm (+7.01mm) | Warmup 역효과, 초기 학습 지연 |
| Attention Lifting v3 | 46.68mm (+5.31mm) | LR 0.001 과다, epoch 2 발산 |
| Attention Lifting v4 | 47.95mm (+6.58mm) | CosineAnnealing이 초기 수렴 방해 |

### 핵심 인사이트

1. **Heatmap의 한계**: Heatmap은 2D 위치의 확률 분포로, 3D depth를 implicit하게만 인코딩 가능
2. **역할 분리 필요**: 2D 위치와 3D depth는 서로 다른 경로로 학습해야 함
3. **Backbone feature 활용**: Backbone의 texture/context 정보가 depth 추정에 핵심
4. **Gradient 흐름 설계**: 3D loss는 backbone으로, heatmap에는 2D loss만

### 다음 실험 계획

| 우선순위 | 실험 | 기대 효과 |
|----------|------|----------|
| 1 | **Attention Lifting v3** | LR=0.001 + MultiStepLR + Gradient Clipping (높은 LR로 빠른 수렴) |
| 2 | AdaIN HeatmapDecoder | Z가 각 Conv layer에 영향, 표현력 향상 |
| 3 | Data Augmentation 강화 | Random rotation, scale, color jitter 등 |

---

## 실행 명령어

```bash
# Baseline (참고용)
python tools/train.py my_code/custom_config/HMD_xregopose_single_coco_full_config.py

# EfficientHeatmapDecoder (완료)
python tools/train.py my_code/custom_config/HMD_xregopose_efficient_decoder_full_config.py
```

---

## 파일 목록

### Head 파일

| Head | 파일 | 용도 | 결과 |
|------|------|------|------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline | **41.37mm 🏆** |
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | + EfficientHeatmapDecoder | 45.06mm |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone | 43.26mm |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup | 45.93mm |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax lifting | 45.92mm |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | Lifting + Backbone | 105.18mm ❌ |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v1 | 45.43mm |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v2 | 48.38mm |

### Config 파일

| Config | 용도 | 결과 |
|--------|------|------|
| `HMD_xregopose_single_coco_full_config.py` | Single COCO baseline | **41.37mm 🏆** |
| `HMD_xregopose_h5cache_coco_mpii_config.py` | Dual COCO+MPII | 43.26mm |
| `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | Dual + Warmup | 45.93mm |
| `HMD_xregopose_single_lifting_config.py` | Lifting only | 45.92mm |
| `HMD_xregopose_lifting_backbone_fusion_config.py` | Lifting + Backbone | 105.18mm ❌ |
| `HMD_xregopose_efficient_decoder_full_config.py` | EfficientHeatmapDecoder | 45.06mm |
| `HMD_xregopose_attention_lifting_full_config.py` | Attention Lifting v1 | 45.43mm |
| `HMD_xregopose_attention_lifting_v2_full_config.py` | Attention Lifting v2 (Warmup+Cosine) | 48.38mm ❌ |
| `HMD_xregopose_attention_lifting_v3_full_config.py` | Attention Lifting v3 (LR=0.001) | 실험 대기 |
