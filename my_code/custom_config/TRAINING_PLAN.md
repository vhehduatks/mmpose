# 훈련 실행 계획

## 현재 상황 (2026-01-20)

### 모델 비교표

| Model | Config | Head | Backbone | Full Body MPJPE |
|-------|--------|------|----------|-----------------|
| **Single COCO (Baseline)** | `HMD_xregopose_single_coco_full_config.py` | `CustomxRegoposeBaselinel1` | ResNet-101 COCO | **42.07mm** 🏆 |
| Dual COCO+MPII | `HMD_xregopose_h5cache_coco_mpii_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | ResNet-101 COCO + MPII | 44.91mm |

### 핵심 구조 차이

| 구분 | Single COCO (42.07mm) | Dual COCO+MPII (44.91mm) |
|------|----------------------|--------------------------|
| **Estimator** | `TopdownPoseEstimator` | `Custom_TopdownPoseEstimator` |
| **Backbone 수** | 1개 (COCO) | 2개 (COCO + MPII) |
| **Mutual Learning** | 없음 | `MSE(feat1, feat2)` |
| **2D→3D 방식** | Heatmap→Encoder→Z[64]→Decoder | 동일 |

**목표**: 42mm 이하 달성

---

## 실험 1: Progressive Warmup (구현 완료 ✅)

**가설**: 초기 epoch에서 pretrained knowledge 보존 → 성능 개선

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

**실행 명령**:
```bash
cd /home/hyeonghwan/github/mmpose
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py
```

**설정**:
- Head: `CustomxRegoposeBaselinel1_multi_backbone_v2`
- mutual_warmup_epochs: 5 (mutual learning 없음)
- mutual_rampup_epochs: 10 (0→1 선형 증가)
- max_epochs: 20

**예상 로그**:
```
[MutualLearningWarmupHook] Epoch 0: mutual_learning_weight = 0.000
[MutualLearningWarmupHook] Epoch 5: mutual_learning_weight = 0.000
[MutualLearningWarmupHook] Epoch 10: mutual_learning_weight = 0.500
[MutualLearningWarmupHook] Epoch 15: mutual_learning_weight = 1.000
```

**Wandb 프로젝트**: `mmpose_xregopose_coco_mpii_warmup`

---

## 실험 2: Ensemble Teacher (미구현)

**가설**: Confidence 기반 앙상블 teacher가 더 나은 knowledge transfer

**구현 필요**:
- `custom_egopose_baselinel1_head_multi_backbone_v3.py`
- Config: `HMD_xregopose_h5cache_coco_mpii_ensemble_config.py`

---

## 실험 3: Heatmap KL Divergence (미구현)

**가설**: MSE 대신 KL divergence가 soft target으로 더 효과적

**구현 필요**:
- `custom_egopose_baselinel1_head_multi_backbone_v4.py`
- Config: `HMD_xregopose_h5cache_coco_mpii_kldiv_config.py`

---

## 실험 4: Single Backbone + Soft-argmax Lifting (구현 완료 ✅)

**가설**: HeatmapEncoder/Decoder 대신 soft-argmax + Lifting으로 구조 단순화 및 성능 개선

### 비교 대상

| 항목 | Single COCO (기존) | Single Lifting (신규) |
|------|-------------------|----------------------|
| **Config** | `HMD_xregopose_single_coco_full_config.py` | `HMD_xregopose_single_lifting_config.py` |
| **Head** | `CustomxRegoposeBaselinel1` | `CustomEgoposeLiftingHead` |
| **Estimator** | `TopdownPoseEstimator` | `TopdownPoseEstimator` |
| **Backbone** | ResNet-101 COCO (동일) | ResNet-101 COCO (동일) |

### 구조 차이

```
[Single COCO - 기존]
Backbone → Heatmap → Encoder → Z[64] → PoseDecoder → 3D
                            ↘ HeatmapDecoder → Recon (40M params!)

[Single Lifting - 신규]
Backbone → Heatmap → soft_argmax → 2D[16,2] + conf[16]
                                        ↓
                              Lifting Network (4M params)
                                        ↓
                                   3D Pose
```

### 장점

| 항목 | 기존 | Lifting |
|------|------|---------|
| Head 파라미터 | ~61M | ~13M |
| HeatmapDecoder | 40M (있음) | 없음 |
| Batch size | 58 | 64 |
| 2D 표현 | Implicit (Z) | Explicit (coords) |
| Loss | Heatmap MSE only | Heatmap MSE + Coord MSE |

### 실행 명령

```bash
cd /home/hyeonghwan/github/mmpose
python tools/train.py my_code/custom_config/HMD_xregopose_single_lifting_config.py
```

**Wandb 프로젝트**: `mmpose_xregopose_single_lifting`

---

## 실행 순서

| 순서 | 실험 | 상태 | Config | 명령어 |
|------|------|------|--------|--------|
| 1 | Progressive Warmup (Dual) | ✅ 준비됨 | `*_warmup_config.py` | `python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py` |
| 2 | **Single Lifting** | ✅ 준비됨 | `*_single_lifting_config.py` | `python tools/train.py my_code/custom_config/HMD_xregopose_single_lifting_config.py` |
| 3 | Ensemble Teacher (Dual) | ❌ 미구현 | - | - |
| 4 | KL Divergence (Dual) | ❌ 미구현 | - | - |

---

## 결과 기록

### 실험 1: Progressive Warmup

| Metric | Value |
|--------|-------|
| Full Body MPJPE | |
| Upper Body MPJPE | |
| Lower Body MPJPE | |
| Best Epoch | |
| Wandb Run | |

**관찰**:
-

**결론**:
-

---

## 결과 기록

### 실험 4: Single Lifting

| Metric | Value |
|--------|-------|
| Full Body MPJPE | |
| Upper Body MPJPE | |
| Lower Body MPJPE | |
| Best Epoch | |
| Wandb Run | |

**비교 (vs Single COCO 42.07mm)**:
-

**관찰**:
-

**결론**:
-

---

## 파일 목록

### 새로 생성된 파일

**Dual Backbone 관련 (Progressive Warmup)**:
- `mmpose/models/heads/heatmap_heads/custom_egopose_baselinel1_head_multi_backbone_v2.py`
- `mmpose/engine/hooks/mutual_learning_hook.py`
- `my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

**Single Backbone + Lifting (신규)**:
- `mmpose/models/heads/heatmap_heads/custom_egopose_lifting_head.py`
- `my_code/custom_config/HMD_xregopose_single_lifting_config.py`

### 수정된 파일
- `mmpose/models/heads/heatmap_heads/__init__.py` (v2, LiftingHead 등록)
