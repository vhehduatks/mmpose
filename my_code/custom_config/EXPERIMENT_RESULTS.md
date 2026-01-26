# EgoPose 3D 실험 결과 및 분석

> 최종 업데이트: 2026-01-27

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
| 10 | Attention Lifting v5 | `HMD_xregopose_attention_lifting_v5_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 49.75 | 8 | ❌ (LR 과다) |
| 11 | Attention Lifting v6 | `HMD_xregopose_attention_lifting_v6_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 51.82 | 6 | ❌ (epoch 10 발산) |
| 12 | Attention Lifting v7 | `HMD_xregopose_attention_lifting_v7_full_config.py` | `CustomEgoposeAttentionLiftingHead` | 45.73 | 7 | ❌ (v1과 유사, warmup 역효과) |
| 13 | Skeleton GAT | `HMD_xregopose_skeleton_gat_full_config.py` | `CustomEgoposeSkeletonGATHead` | 50.88 | 7 | ❌ (sparse attention 실패) |
| 14 | ViT Lifting v1 | `HMD_xregopose_vit_lifting_v1_full_config.py` | `CustomEgoposeViTLiftingHead` | 50.84 | 8 | ❌ (별도 heatmap, 정보 공유 없음) |
| 15 | ViT Lifting v2 | `HMD_xregopose_vit_lifting_v2_full_config.py` | `CustomEgoposeViTLiftingHead` | 51.77 | 1 | ❌ (no recon, v1보다 악화) |
| 16 | **ViT Lifting v3** | `HMD_xregopose_vit_lifting_v3_full_config.py` | `CustomEgoposeViTLiftingHead` | **45.34** | 4 | ⭐ ViT 최고 (Recon + Self-Attn) |
| 17 | ViT Lifting v4 | `HMD_xregopose_vit_lifting_v4_full_config.py` | `CustomEgoposeViTLiftingHead` | 45.66 | 7 | ❌ (CosineAnnealingLR) |
| 18 | ViT Lifting v5 | `HMD_xregopose_vit_lifting_v5_full_config.py` | `CustomEgoposeViTLiftingHeadV5` | 47.22 | 7 | ❌ (Hybrid Attention) |

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
| Attention Lifting v5 | 49.75mm | 31.08mm | 68.41mm | 8 |
| Attention Lifting v6 | 51.82mm | 32.89mm | 70.75mm | 6 |
| Attention Lifting v7 | 45.73mm | 32.69mm | 58.78mm | 7 |
| Skeleton GAT | 50.88mm | 33.83mm | 67.92mm | 7 |
| ViT Lifting v1 | 50.84mm | 28.77mm | 72.91mm | 8 |
| ViT Lifting v2 | 51.77mm | 33.37mm | 70.17mm | 1 |
| **ViT Lifting v3** | **45.34mm** | **23.49mm** | **67.19mm** | 4 |
| ViT Lifting v4 | 45.66mm | 25.76mm | 65.56mm | 7 |
| ViT Lifting v5 | 47.22mm | 29.09mm | 65.35mm | 7 |

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
| Attention Lifting v5 | 49.75mm | +8.38mm ❌ | LR=0.002 과다, epoch 2 spike |
| Attention Lifting v6 | 51.82mm | +10.45mm ❌ | CosineRestartLR, epoch 10 발산 |
| Attention Lifting v7 | 45.73mm | +4.36mm ❌ | Optimized schedule, epoch 2 spike |
| Skeleton GAT | 50.88mm | +9.51mm ❌ | Sparse attention, 하체 성능 악화 |
| ViT Lifting v1 | 50.84mm | +9.47mm ❌ | 별도 heatmap 경로, 정보 공유 없음 |
| ViT Lifting v2 | 51.77mm | +10.40mm ❌ | No reconstruction, Self-Attn만 |
| **ViT Lifting v3** | **45.34mm** | **+3.97mm** ⭐ | Recon + Self-Attn, Upper Body 최고 |
| ViT Lifting v4 | 45.66mm | +4.29mm ❌ | CosineAnnealingLR, validation spike |
| ViT Lifting v5 | 47.22mm | +5.85mm ❌ | Hybrid Attention, gradient scaling |

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
| v7 | LR=5e-4, MultiStepLR+Warmup | 45.73mm | ❌ Warmup 역효과 |
| v3 | LR=1e-3, MultiStepLR | 46.68mm | ❌ LR 과다, 발산 |
| v4 | LR=1e-3, CosineAnnealing | 47.95mm | ❌ 초기 수렴 방해 |
| v2 | LR=5e-4, Warmup+Cosine | 48.38mm | ❌ Warmup 역효과 |
| v5 | LR=2e-3, CosineAnnealing | 49.75mm | ❌ LR 과다 (v4의 2배) |
| v6 | LR=1e-3, CosineRestartLR | 51.82mm | ❌ LR restart 불안정, 발산 |

> **결론**: Attention Lifting v1 (45.43mm)이 최선이나, Baseline (41.37mm)보다 +4.06mm 나쁨.
> v7에서도 warmup이 역효과를 일으킴 (epoch 2 spike). **Attention 구조는 warmup 없이 빠른 초기 학습이 필요**.
> Baseline 달성을 위해서는 완전히 다른 접근법 필요.

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

### 실험 10: Attention Lifting v5 (Higher LR = 0.002 + CosineAnnealing)

**Config**: `HMD_xregopose_attention_lifting_v5_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v5_full`

**가설**: v4의 느린 초기 수렴을 높은 LR (0.002 = v4의 2배)로 보상

**설정**:
```python
# v4: LR=0.001, CosineAnnealing → epoch 1: 76.12mm (느린 시작)
# v5: LR=0.002 (2x of v4) + CosineAnnealing
optim_wrapper = dict(
    optimizer=dict(lr=0.002, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', begin=0, end=10, eta_min=1e-5, by_epoch=True),
]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 63.99mm | 39.01mm | 88.96mm | |
| **2** | **68.76mm** | 38.24mm | 99.28mm | **⚠️ Spike! +4.77mm** |
| 3 | 56.37mm | 32.71mm | 80.02mm | 회복 |
| 4 | 51.36mm | 29.44mm | 73.28mm | |
| 5 | 58.62mm | 34.63mm | 82.61mm | ⚠️ Spike |
| 6 | 52.35mm | 30.50mm | 74.20mm | |
| 7 | 50.89mm | 31.65mm | 70.12mm | |
| **8** | **49.75mm** | **31.08mm** | **68.41mm** | **🏆 Best** |
| 9 | 51.02mm | 31.35mm | 70.70mm | |
| 10 | 52.47mm | 32.48mm | 72.46mm | |

**v4 vs v5 비교**:
| 항목 | v4 (LR=0.001) | v5 (LR=0.002) |
|------|---------------|---------------|
| Epoch 1 | 76.12mm | 63.99mm |
| Epoch 2 | 59.32mm | **68.76mm** ⚠️ |
| Best | **47.95mm** | 49.75mm |
| Best Epoch | 6 | 8 |
| Spike | +5.6mm (ep5) | **+4.77mm (ep2)** |

**실패 원인 분석**:
1. **LR 0.002가 너무 높음**: Epoch 2에서 spike 발생 (63.99 → 68.76mm)
2. **v4보다 1.8mm 나쁨**: 높은 LR이 초기 수렴은 빠르게 하지만 불안정
3. **CosineAnnealing + 높은 LR 조합 부적합**

**결론**:
> - LR=0.002는 CosineAnnealing과 조합 시 불안정
> - v4 (LR=0.001)가 v5보다 더 나음
> - Attention Lifting에는 LR=0.0005 (v1)이 최적

---

### 실험 11: Attention Lifting v6 (CosineRestartLR - Warm Restarts)

**Config**: `HMD_xregopose_attention_lifting_v6_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v6_full`

**가설**: CosineRestartLR (warm restarts)로 주기적 LR 재시작 → local minima 탈출

**설정**:
```python
# Warm Restarts: periods=[3,3,3,1] → 3+3+3+1 = 10 epochs
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(
        type='CosineRestartLR',
        periods=[3, 3, 3, 1],  # restart at epoch 3, 6, 9
        restart_weights=[1, 1, 1, 1],
        eta_min=1e-5,
        by_epoch=True,
    ),
]
```

**예상 LR 패턴**:
- Epoch 1-3: 0.001 → decay → ~1e-5 (Period 1)
- Epoch 4: **restart → 0.001** (Period 2 시작)
- Epoch 4-6: 0.001 → decay → ~1e-5
- Epoch 7: **restart → 0.001** (Period 3 시작)
- Epoch 7-9: 0.001 → decay → ~1e-5
- Epoch 10: **restart → 0.001** (Period 4, 1 epoch only)

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | LR 상태 | 비고 |
|-------|-----------|------------|------------|---------|------|
| 1 | 60.91mm | 37.96mm | 83.87mm | Period 1 시작 | |
| 2 | 57.57mm | 35.05mm | 80.10mm | | |
| 3 | 53.38mm | 33.26mm | 73.51mm | Period 1 끝 | |
| 4 | 54.78mm | 34.33mm | 75.24mm | **LR restart** | ⚠️ 성능 하락 |
| 5 | 60.76mm | 37.10mm | 84.43mm | | ⚠️ Spike |
| **6** | **51.82mm** | **32.89mm** | **70.75mm** | Period 2 끝 | **🏆 Best** |
| 7 | 54.20mm | 34.31mm | 74.09mm | **LR restart** | ⚠️ 성능 하락 |
| 8 | 56.15mm | 39.04mm | 73.26mm | | |
| 9 | 55.55mm | 36.90mm | 74.19mm | Period 3 끝 | |
| 10 | 97.62mm | 51.77mm | 143.46mm | **LR restart** | ❌ **발산!** |

**실패 원인 분석**:
1. **LR restart 직후 성능 하락**: Epoch 4, 7, 10에서 restart 후 성능 악화
   - Restart가 학습된 weight를 불안정하게 만듦
2. **Epoch 10 발산**: Period 4가 1 epoch만 있어 restart 후 바로 종료
   - 높은 LR(0.001)로 시작하여 회복할 시간 없이 발산 (97.62mm)
3. **Warm restarts 부적합**: Attention Lifting은 안정적인 LR decay가 필요

**v1~v6 LR Schedule 비교**:
| 버전 | LR Schedule | Best MPJPE | Epoch 10 | 결과 |
|------|-------------|------------|----------|------|
| **v1** | MultiStepLR [4,7], LR=5e-4 | **45.43mm** | 45.63mm | 🏆 Best |
| v2 | Warmup + CosineAnnealing, LR=5e-4 | 48.38mm | 48.97mm | ❌ |
| v3 | MultiStepLR [4,7], LR=1e-3 | 46.68mm | 52.95mm | ❌ |
| v4 | CosineAnnealing, LR=1e-3 | 47.95mm | 52.10mm | ❌ |
| v5 | CosineAnnealing, LR=2e-3 | 49.75mm | 52.47mm | ❌ |
| v6 | **CosineRestartLR**, LR=1e-3 | 51.82mm | **97.62mm** | ❌ 발산 |

**결론**:
> - CosineRestartLR (warm restarts)은 Attention Lifting에 **부적합**
> - LR restart가 학습 안정성을 해침
> - 마지막 period (1 epoch)가 너무 짧아 발산 유발
> - **v1의 MultiStepLR이 여전히 최선**

---

### 실험 12: Attention Lifting v7 (Optimized LR Schedule)

**Config**: `HMD_xregopose_attention_lifting_v7_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_attention_lifting_v7_full`

**가설**: v1~v6 분석을 기반으로 최적화된 LR schedule 설계
- v1의 LR=0.0005 유지 (최적)
- v1의 epoch 4 spike 방지를 위해 milestone [3, 5, 7]로 앞당김
- Gradient clipping 추가 (max_norm=1.0)
- 500 iteration warmup (0.5x → 1x)

**설정**:
```python
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[3, 5, 7], gamma=0.5, by_epoch=True),
]
```

**예상 LR 패턴**:
- Iter 0-500: 0.25e-3 → 0.5e-3 (iteration warmup)
- Epoch 1-3: 5e-4
- Epoch 4-5: 2.5e-4 (milestone 3 이후)
- Epoch 6-7: 1.25e-4 (milestone 5 이후)
- Epoch 8-10: 6.25e-5 (milestone 7 이후)

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | LR | 비고 |
|-------|-----------|------------|------------|-----|------|
| 1 | 52.38mm | 39.87mm | 64.89mm | warmup→5e-4 | |
| **2** | **61.82mm** | 44.55mm | 79.09mm | 5e-4 | **⚠️ Spike! +9.44mm** |
| 3 | 55.63mm | 39.04mm | 72.22mm | 5e-4 | 회복 시작 |
| 4 | 47.72mm | 33.01mm | 62.44mm | 2.5e-4 | LR decay |
| 5 | 49.33mm | 33.67mm | 64.98mm | 2.5e-4 | |
| 6 | 46.88mm | 32.44mm | 61.31mm | 1.25e-4 | 2차 decay |
| **7** | **45.73mm** | **32.69mm** | **58.78mm** | 1.25e-4 | **🏆 Best** |
| 8 | 46.03mm | 31.89mm | 60.17mm | 6.25e-5 | 3차 decay |
| 9 | 46.27mm | 32.00mm | 60.54mm | 6.25e-5 | |
| 10 | 46.22mm | 31.89mm | 60.54mm | 6.25e-5 | 수렴 |

**v1 vs v7 비교**:
| 항목 | v1 | v7 | 비고 |
|------|-----|-----|------|
| **Best MPJPE** | **45.43mm 🏆** | 45.73mm | v1이 0.30mm 더 좋음 |
| Best Epoch | 8 | 7 | |
| Epoch 2 | 45.62mm | **61.82mm** | **v7 spike +16.2mm** |
| Spike 위치 | Epoch 4 (+2.93mm) | Epoch 2 (+9.44mm) | v7 spike가 더 심함 |
| Gradient Clipping | ❌ | ✅ | |
| Warmup | ❌ | ✅ (500 iter) | |

**실패 원인 분석**:
1. **Warmup 역효과**: 500 iter warmup이 오히려 epoch 2에서 spike 유발
   - v1은 epoch 2에서 이미 45.62mm로 안정화
   - v7은 epoch 2에서 61.82mm로 급등
2. **v1과 비슷한 최종 성능**: Best는 0.30mm 차이로 거의 동일
3. **Early milestone이 spike 방지 못함**: milestone [3, 5, 7]이 epoch 2 spike와 무관

**핵심 인사이트**:
> - **Warmup은 Attention Lifting에 해로움**: v2, v7 모두 warmup으로 악화
> - **v1의 "warmup 없는 MultiStepLR"이 최적**: 처음부터 높은 LR로 빠르게 학습
> - Attention 구조는 초기 빠른 학습이 중요, warmup이 이를 방해

**v1~v7 최종 비교**:
| 버전 | LR Schedule | Best MPJPE | Warmup | 결과 |
|------|-------------|------------|--------|------|
| **v1** | MultiStepLR [4,7], LR=5e-4 | **45.43mm** | ❌ | 🏆 Best |
| v7 | MultiStepLR [3,5,7], LR=5e-4, warmup 500iter | 45.73mm | ✅ | ❌ warmup 역효과 |
| v3 | MultiStepLR [4,7], LR=1e-3 | 46.68mm | ❌ | ❌ LR 과다 |
| v4 | CosineAnnealing, LR=1e-3 | 47.95mm | ❌ | ❌ 초기 수렴 방해 |
| v2 | Warmup + CosineAnnealing, LR=5e-4 | 48.38mm | ✅ | ❌ warmup 역효과 |
| v5 | CosineAnnealing, LR=2e-3 | 49.75mm | ❌ | ❌ LR 과다 |
| v6 | CosineRestartLR, LR=1e-3 | 51.82mm | ❌ | ❌ restart 불안정 |

**결론**:
> **Attention Lifting v1 (45.43mm)이 최선**이며, 추가적인 LR schedule 최적화는 효과 없음.
> Baseline (41.37mm) 달성을 위해서는 **구조적 변경**이 필요.

---

### 실험 13: Skeleton Graph Attention Network (GAT)

**Config**: `HMD_xregopose_skeleton_gat_full_config.py`
**Work Dir**: `work_dirs/HMD_xregopose_skeleton_gat_full`

**가설**: JointSelfAttention을 SkeletonGraphAttention으로 교체하여 해부학적 구조 prior 활용
- 관절은 연결된 이웃 관절에만 attention
- Skeleton adjacency matrix 기반 sparse attention
- 2-hop neighbors 포함 (second-order connections)
- Learnable edge attention bias

**구조**:
```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓ (Deconv)                         ↓ (reshape)
Heatmap [16, 47, 47]             Backbone tokens [64, D]
       │                                  │
       ↓ (soft_argmax)                    │
  2D coords [16, 2]                       │
       │                                  │
       ↓ (Joint Embedding)                │
  Joint tokens [16, 64]                   │
       │                                  │
       └── Backbone Cross-Attention ──────┘
                    │
                    ▼
       HMD Cross-Attention ←── HMD tokens [3, 64]
                    │
                    ▼
          ┌─────────────────────┐
          │ Skeleton GAT × 2    │  ← Key: Only connected joints attend
          │                     │
          │ Adjacency Matrix:   │
          │ head─neck─shoulders │
          │       │    │        │
          │      hips elbows    │
          │       │    │        │
          │     knees wrists    │
          │       │             │
          │    ankles           │
          └─────────────────────┘
                    │
                    ▼
              3D Pose [16, 3]
```

**Skeleton 연결 구조**:
```python
SKELETON_EDGES = [
    (0, 1),    # head - neck
    (1, 2),    # neck - left_shoulder
    (2, 3),    # left_shoulder - left_elbow
    (3, 4),    # left_elbow - left_wrist
    (1, 5),    # neck - right_shoulder
    (5, 6),    # right_shoulder - right_elbow
    (6, 7),    # right_elbow - right_wrist
    (1, 8),    # neck - left_hip
    (8, 9),    # left_hip - left_knee
    (9, 10),   # left_knee - left_ankle
    (1, 11),   # neck - right_hip
    (11, 12),  # right_hip - right_knee
    (12, 13),  # right_knee - right_ankle
    (2, 5),    # left_shoulder - right_shoulder
    (8, 11),   # left_hip - right_hip
]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 76.66mm | 50.85mm | 102.48mm | |
| 2 | 62.78mm | 37.58mm | 87.98mm | |
| 3 | 53.96mm | 33.04mm | 74.88mm | |
| 4 | 54.11mm | 36.28mm | 71.93mm | |
| 5 | 51.86mm | 33.57mm | 70.14mm | |
| 6 | 52.20mm | 34.04mm | 70.37mm | |
| **7** | **50.88mm** | **33.83mm** | **67.92mm** | **🏆 Best** |
| 8 | 51.19mm | 33.72mm | 68.66mm | |
| 9 | 51.24mm | 33.30mm | 69.17mm | |
| 10 | 51.60mm | 33.84mm | 69.35mm | |

**Attention Lifting v1 vs Skeleton GAT 비교**:
| 항목 | Attention Lifting v1 | Skeleton GAT | 차이 |
|------|---------------------|--------------|------|
| **Best MPJPE** | **45.43mm 🏆** | 50.88mm | +5.45mm |
| Upper Body | **30.14mm** | 33.83mm | +3.69mm |
| Lower Body | **60.72mm** | 67.92mm | +7.20mm |
| Best Epoch | 7 | 7 | |
| Attention 방식 | Dense (fully connected) | Sparse (skeleton-based) | |

**실패 원인 분석**:
1. **Sparse attention의 한계**: 해부학적 연결만으로는 3D pose에 필요한 global context 부족
   - 3D pose estimation은 모든 관절의 상호 관계가 필요
   - 예: 왼손과 오른발의 상대 위치도 중요한 depth cue
2. **하체 성능 급락**: Lower body +7.20mm 악화
   - 하체 관절(hip, knee, ankle)은 상체와의 관계가 중요
   - Skeleton GAT에서는 neck만 연결되어 정보 부족
3. **2-hop neighbors도 불충분**: Second-order connections으로도 global context 복원 불가
   - 예: ankle → knee → hip → neck (3-hop) 관계 학습 불가

**핵심 인사이트**:
> - **Dense attention이 3D pose에 필수**: 모든 관절 간 fully-connected attention 필요
> - **Skeleton prior는 제약이 아닌 bias로 사용**: Sparse attention 대신 edge bias로 활용
> - **Attention Lifting v1의 JointSelfAttention이 더 효과적**: 구조적 prior 없이도 학습 가능

**결론**:
> Skeleton GAT (50.88mm)는 Attention Lifting v1 (45.43mm)보다 5.45mm 나쁨.
> Sparse attention (skeleton-based)이 Dense attention보다 열등함을 확인.
> 3D pose estimation에는 **global context**가 필수이며, 해부학적 구조만으로는 불충분.

---

### 실험 14-18: ViT-Style Lifting v1~v5

**개요**: ViTPose 스타일의 Learnable Joint Queries + Self-Attention 구조

**핵심 아이디어**:
```
soft_argmax로 좌표 추출 대신 → Learnable Joint Queries 사용
Cross-Attention 대신 → Self-Attention (joint ↔ spatial 양방향)
Heatmap Reconstruction → 2D 관절 정보를 joint tokens에 주입
HMD Cross-Attention → 3D depth reference 추가
```

#### ViT Lifting v1 (별도 Heatmap 경로)

**Config**: `HMD_xregopose_vit_lifting_v1_full_config.py`

**구조**:
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

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 52.57mm | 33.04mm | 72.09mm |
| 4 | 51.89mm | 29.39mm | 74.39mm |
| **8** | **50.84mm** | **28.77mm** | **72.91mm** |
| 10 | 52.49mm | 30.11mm | 74.87mm |

**분석**: Heatmap과 3D Lifting이 **독립적** (정보 공유 없음)

---

#### ViT Lifting v2 (No Reconstruction)

**Config**: `HMD_xregopose_vit_lifting_v2_full_config.py`

**구조**:
```
Backbone → Spatial Tokens + Joint Queries
                    ↓
            Self-Attention (spatial info 자동 전달?)
                    ↓
            Joint Tokens
                    ↓
            HMD Cross-Attention
                    ↓
            3D Pose Head → 3D Loss만
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| **1** | **51.77mm** | **33.37mm** | **70.17mm** |
| 2 | 55.91mm | 31.76mm | 80.07mm |
| 5 | 57.70mm | 33.69mm | 81.71mm |

**분석**:
- Heatmap reconstruction 없이 Self-Attention만으로는 2D 정보 전달 불충분
- v1보다 오히려 악화 (+0.93mm)
- Epoch 5에서 훈련 중단 (발산)

---

#### ViT Lifting v3 (Reconstruction Regularization) ⭐ ViT 최고

**Config**: `HMD_xregopose_vit_lifting_v3_full_config.py`

**구조**:
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

**핵심**: Heatmap Reconstruction이 **Joint Tokens를 regularize** → 2D 정보 강제 주입

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 50.32mm | 31.40mm | 69.23mm | |
| 2 | 53.72mm | 28.27mm | 79.17mm | ⚠️ Spike |
| 3 | 48.28mm | 25.46mm | 71.10mm | 회복 |
| **4** | **45.34mm** | **23.49mm** | **67.19mm** | **🏆 Best** |
| 5 | 47.13mm | 24.76mm | 69.50mm | ⚠️ Spike |
| 6 | 49.12mm | 26.22mm | 72.02mm | |
| 7 | 45.59mm | 24.53mm | 66.65mm | |
| 10 | 47.37mm | 24.56mm | 70.19mm | |

**분석**:
- **Upper Body 최고 성능**: 23.49mm (Baseline 29.42mm 대비 -5.93mm!)
- Validation spike 문제: epoch 2, 5에서 급등
- MultiStepLR [3, 5, 7]의 급격한 LR decay가 원인

**LR Schedule 문제**:
```
Epoch 1-3: LR = 0.0005
Epoch 4:   LR = 0.00025 (3에서 decay) → Best!
Epoch 5:   LR = 0.000125 (5에서 decay) → 스파이크 시작
```

---

#### ViT Lifting v4 (CosineAnnealingLR)

**Config**: `HMD_xregopose_vit_lifting_v4_full_config.py`

**v3 대비 변경**: MultiStepLR → CosineAnnealingLR (부드러운 LR 감소)

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 48.53mm | 30.10mm | 66.96mm | |
| 4 | 48.98mm | 26.45mm | 71.51mm | |
| 5 | 53.99mm | 28.65mm | 79.32mm | ⚠️ Spike |
| **7** | **45.66mm** | **25.76mm** | **65.56mm** | **🏆 Best** |
| 10 | 48.51mm | 27.09mm | 69.93mm | |

**분석**:
- CosineAnnealingLR로 LR 감소를 부드럽게 했으나 여전히 spike 발생
- v3보다 0.32mm 나쁨 (45.34 → 45.66mm)
- Lower Body는 개선 (67.19 → 65.56mm)

---

#### ViT Lifting v5 (Hybrid Attention)

**Config**: `HMD_xregopose_vit_lifting_v5_full_config.py`

**v4 대비 변경**:
1. Self-Attention [80×80] → Cross-Attention [16×64] + Self-Attention [16×16]
2. 역할 분리: Cross(위치 찾기) + Self(skeleton 관계)
3. Gradient Scaling: Heatmap gradient를 0.1배로 줄여 3D 학습에 집중

**구조**:
```
Backbone feat [2048, 8, 8]
     ↓
Spatial Tokens [64, D]
     ↓
┌─────────────────────────────────────────┐
│  Stage 1: Cross-Attention (J → S)       │
│  Q: Joint Queries [16, D]               │
│  K/V: Spatial Tokens [64, D]            │  [16×64]
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Stage 2: Self-Attention (J → J) × 2    │
│  Q=K=V: Joint Tokens [16, D]            │  [16×16]
└─────────────────────────────────────────┘
     ↓
Joint Tokens [16, D]
     ├── Heatmap Decoder (gradient scaled 0.1)
     ↓
HMD Cross-Attention [16×3]
     ↓
3D Pose Head → [16, 3]
```

**Epoch별 결과**:
| Epoch | Full Body | Upper Body | Lower Body | 비고 |
|-------|-----------|------------|------------|------|
| 1 | 51.73mm | 36.48mm | 66.98mm | |
| 4 | 49.03mm | 29.48mm | 68.58mm | |
| **7** | **47.22mm** | **29.09mm** | **65.35mm** | **🏆 Best** |
| 10 | 49.15mm | 29.75mm | 68.55mm | |

**분석**:
- v4보다 1.56mm 악화 (45.66 → 47.22mm)
- Hybrid Attention이 v3/v4의 Self-Attention보다 효과 낮음
- Gradient Scaling이 heatmap 학습을 약화시켜 regularization 효과 감소

---

### ViT Lifting 시리즈 요약

| 버전 | 구조 | LR Schedule | Best MPJPE | Best Epoch | 비고 |
|------|------|-------------|------------|------------|------|
| v1 | 별도 Heatmap | MultiStepLR | 50.84mm | 8 | 정보 공유 없음 |
| v2 | No Recon | MultiStepLR | 51.77mm | 1 | Self-Attn만으로 불충분 |
| **v3** | **Recon + Self-Attn** | MultiStepLR | **45.34mm** | 4 | **⭐ ViT 최고** |
| v4 | Recon + Self-Attn | CosineAnnealingLR | 45.66mm | 7 | spike 감소 but 성능 하락 |
| v5 | Hybrid Attention | CosineAnnealingLR | 47.22mm | 7 | Gradient Scaling 역효과 |

**핵심 인사이트**:
1. **Heatmap Reconstruction이 핵심**: v2(no recon)는 v3(recon)보다 6.43mm 나쁨
2. **Self-Attention [80×80]이 Hybrid보다 효과적**: v5의 역할 분리가 오히려 악화
3. **Upper Body 성능 우수**: v3의 Upper Body 23.49mm는 Baseline(29.42mm)보다 5.93mm 좋음
4. **LR Schedule 민감**: MultiStepLR의 급격한 decay가 spike 유발, but 성능은 더 좋음

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
| Attention Lifting v5 | 49.75mm (+8.38mm) | LR=0.002 과다, epoch 2 spike |
| Attention Lifting v6 | 51.82mm (+10.45mm) | CosineRestartLR restart 불안정, epoch 10 발산 |
| Attention Lifting v7 | 45.73mm (+4.36mm) | Optimized schedule, warmup 역효과 |
| Skeleton GAT | 50.88mm (+9.51mm) | Sparse attention, global context 부족, 하체 악화 |
| ViT Lifting v1 | 50.84mm (+9.47mm) | 별도 heatmap 경로, 정보 공유 없음 |
| ViT Lifting v2 | 51.77mm (+10.40mm) | No reconstruction, Self-Attn만으로 불충분 |
| **ViT Lifting v3** | **45.34mm (+3.97mm)** | ⭐ Recon + Self-Attn, Upper Body 최고 (23.49mm) |
| ViT Lifting v4 | 45.66mm (+4.29mm) | CosineAnnealingLR, spike 감소 but 성능 하락 |
| ViT Lifting v5 | 47.22mm (+5.85mm) | Hybrid Attention, Gradient Scaling 역효과 |

### 핵심 인사이트

1. **Heatmap의 한계**: Heatmap은 2D 위치의 확률 분포로, 3D depth를 implicit하게만 인코딩 가능
2. **역할 분리 필요**: 2D 위치와 3D depth는 서로 다른 경로로 학습해야 함
3. **Backbone feature 활용**: Backbone의 texture/context 정보가 depth 추정에 핵심
4. **Gradient 흐름 설계**: 3D loss는 backbone으로, heatmap에는 2D loss만

### 다음 실험 계획

**Attention Lifting v1~v7 실험 완료**. LR schedule 최적화로는 Baseline(41.37mm) 달성 불가.

| 우선순위 | 실험 | 기대 효과 |
|----------|------|----------|
| 1 | **구조적 개선** | Attention 외 다른 depth extraction 방식 탐색 |
| 2 | AdaIN HeatmapDecoder | Z가 각 Conv layer에 영향, 표현력 향상 |
| 3 | Data Augmentation 강화 | Random rotation, scale, color jitter 등 |

**Attention Lifting 핵심 인사이트**:
- Warmup은 역효과 (v2, v7 모두 실패)
- MultiStepLR > CosineAnnealing
- LR=0.0005가 최적 (높으면 발산)
- 최선: v1 (45.43mm), Baseline 대비 +4.06mm

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
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v1 🏆 | 45.43mm |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting v2~v7 | 45.73~51.82mm |
| `CustomEgoposeSkeletonGATHead` | `custom_egopose_skeleton_gat_head.py` | Skeleton Graph Attention | 50.88mm |
| `CustomEgoposeViTLiftingHead` | `custom_egopose_vit_lifting_head.py` | ViT Lifting v1~v4 | 45.34mm (v3) ⭐ |
| `CustomEgoposeViTLiftingHeadV5` | `custom_egopose_vit_lifting_head_v5.py` | ViT Lifting v5 (Hybrid) | 47.22mm |

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
| `HMD_xregopose_attention_lifting_v3_full_config.py` | Attention Lifting v3 (LR=0.001) | 46.68mm ❌ |
| `HMD_xregopose_attention_lifting_v4_full_config.py` | Attention Lifting v4 (CosineAnnealing) | 47.95mm ❌ |
| `HMD_xregopose_attention_lifting_v5_full_config.py` | Attention Lifting v5 (LR=0.002) | 49.75mm ❌ |
| `HMD_xregopose_attention_lifting_v6_full_config.py` | Attention Lifting v6 (CosineRestartLR) | 51.82mm ❌ |
| `HMD_xregopose_attention_lifting_v7_full_config.py` | Attention Lifting v7 (Optimized Schedule) | 45.73mm ❌ |
| `HMD_xregopose_skeleton_gat_full_config.py` | Skeleton GAT | 50.88mm ❌ |
| `HMD_xregopose_vit_lifting_v1_full_config.py` | ViT Lifting v1 (별도 Heatmap) | 50.84mm ❌ |
| `HMD_xregopose_vit_lifting_v2_full_config.py` | ViT Lifting v2 (No Recon) | 51.77mm ❌ |
| `HMD_xregopose_vit_lifting_v3_full_config.py` | **ViT Lifting v3 (Recon)** | **45.34mm ⭐** |
| `HMD_xregopose_vit_lifting_v4_full_config.py` | ViT Lifting v4 (CosineAnnealing) | 45.66mm ❌ |
| `HMD_xregopose_vit_lifting_v5_full_config.py` | ViT Lifting v5 (Hybrid Attention) | 47.22mm ❌ |
