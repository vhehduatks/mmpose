# EgoPose 3D 실험 결과 및 분석

> 최종 업데이트: 2026-01-21

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

### 부위별 결과 상세

| 실험명 | Full Body | Upper Body | Lower Body | Best Epoch |
|--------|-----------|------------|------------|------------|
| **Single COCO (Baseline)** | **41.37mm** | **29.42mm** | **53.31mm** | 8 |
| Dual COCO+MPII | 43.26mm | 30.03mm | 56.48mm | 8 |
| Dual Warmup v2 | 45.93mm | 31.07mm | 60.79mm | 9 |
| Single Lifting | 45.92mm | 33.91mm | 57.93mm | 9 |

### Baseline 대비 비교

| 실험명 | Full Body | vs Baseline | 비고 |
|--------|-----------|-------------|------|
| **Single COCO (Baseline)** | **41.37mm** | - | 🏆 |
| Dual COCO+MPII | 43.26mm | +1.89mm ❌ | mutual learning 악화 |
| Dual Warmup v2 | 45.93mm | +4.56mm ❌ | warmup도 효과 없음 |
| Single Lifting | 45.92mm | +4.55mm ❌ | depth 정보 부족 |

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

**상태**: 🔄 구현 완료, 실험 대기

---

## 종합 분석

### 실패한 접근법

| 접근법 | 결과 | 문제점 |
|--------|------|--------|
| Dual Backbone + Mutual Learning | 43.26mm (+1.89mm) | COCO/MPII feature 분포 충돌 |
| Progressive Warmup | 45.93mm (+4.56mm) | 근본적 mutual learning 문제 해결 불가 |
| Pure 2D→3D Lifting | 45.92mm (+4.55mm) | Depth 정보 부재, 학습 불안정 |

### 핵심 인사이트

1. **Heatmap의 한계**: Heatmap은 2D 위치의 확률 분포로, 3D depth를 implicit하게만 인코딩 가능
2. **역할 분리 필요**: 2D 위치와 3D depth는 서로 다른 경로로 학습해야 함
3. **Backbone feature 활용**: Backbone의 texture/context 정보가 depth 추정에 핵심
4. **Gradient 흐름 설계**: 3D loss는 backbone으로, heatmap에는 2D loss만

### 다음 실험 계획

| 우선순위 | 실험 | 기대 효과 |
|----------|------|----------|
| 1 | **Lifting + Backbone Fusion (5-C)** | 역할 분리로 depth 정보 활용 |
| 2 | Attention Lifting (5-D) | 관절별 selective depth query |

---

## 실행 명령어

```bash
# Baseline (참고용)
python tools/train.py my_code/custom_config/HMD_xregopose_single_coco_full_config.py

# Lifting + Backbone Fusion (다음 실험)
python tools/train.py my_code/custom_config/HMD_xregopose_lifting_backbone_fusion_config.py
```

---

## 파일 목록

### Head 파일

| Head | 파일 | 용도 | 결과 |
|------|------|------|------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline | **41.37mm 🏆** |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone | 43.26mm |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup | 45.93mm |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax lifting | 45.92mm |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | **Lifting + Backbone** | 실험 대기 |

### Config 파일

| Config | 용도 | 결과 |
|--------|------|------|
| `HMD_xregopose_single_coco_full_config.py` | Single COCO baseline | **41.37mm 🏆** |
| `HMD_xregopose_h5cache_coco_mpii_config.py` | Dual COCO+MPII | 43.26mm |
| `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | Dual + Warmup | 45.93mm |
| `HMD_xregopose_single_lifting_config.py` | Lifting only | 45.92mm |
| `HMD_xregopose_lifting_backbone_fusion_config.py` | **Lifting + Backbone** | 실험 대기 |
