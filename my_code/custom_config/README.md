# XR EgoPose Custom Configs

Egocentric 3D pose estimation을 위한 커스텀 설정 파일들입니다.

## 모델 비교

| Config | Head Type | 특징 |
|--------|-----------|------|
| `HMD_xregopose_h5cache_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | Baseline (H5 캐시) |
| `HMD_xregopose_h5cache_augmented_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | + Data Augmentation + Deeper Decoder |
| `HMD_xregopose_confidence_weighted_config.py` | `ConfidenceWeightedHMDHead` | + Confidence-weighted HMD Fusion |

## 모델 개선사항

### 1. Data Augmentation (augmented_config)

색상 기반 augmentation만 적용 (3D 좌표 호환성 유지):

```python
dict(
    type='PhotometricDistortion',
    brightness_delta=32,
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_delta=15,
)
```

> **Note**: RandomFlip, RandomBBoxTransform은 3D GT 좌표 변환이 필요하여 제외

### 2. Pose Decoder Depth (augmented_config)

Residual block 수 증가:

| 설정 | num_stage | Parameters |
|------|-----------|------------|
| 기존 | 1 | 586K |
| 개선 | 2 | 1.1M |

```python
head=dict(
    pose_decoder_num_stage=2,
    pose_decoder_linear_size=512,
    pose_decoder_dropout=0.3,
)
```

### 3. Confidence-Weighted HMD Fusion (confidence_weighted_config)

히트맵 신뢰도 기반 동적 가중치:

```
┌─────────────┐     ┌─────────────┐
│  Heatmap    │     │  HMD Info   │
│  (16, 47,47)│     │    (9,)     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌─────────────┐
│   Encoder    │    │  Linear     │
│  → z (64)    │    │  → z (64)   │
└──────┬───────┘    └──────┬──────┘
       │                   │
       │    ┌──────────┐   │
       └───►│ Confidence│◄──┘
            │  Fusion   │
            └─────┬─────┘
                  │
                  ▼
            ┌───────────┐
            │  z_fused  │
            │   (64)    │
            └─────┬─────┘
                  │
                  ▼
            ┌───────────┐
            │   Pose    │
            │  Decoder  │
            └─────┬─────┘
                  │
                  ▼
            ┌───────────┐
            │  3D Pose  │
            │ (16, 3)   │
            └───────────┘
```

**동작 원리:**
```python
# 히트맵에서 관절별 신뢰도 추출
confidence = sigmoid(heatmap.max(dim=(-1,-2)))  # [B, 16]

# 신뢰도 → 가중치 (학습 가능한 projection)
visual_weight = confidence_proj(confidence)     # [B, 64]
hmd_weight = 1 - visual_weight

# 가중 융합
z_fused = z_visual * visual_weight + z_hmd * hmd_weight
```

**기대 효과:**
- 보이는 관절 (high confidence) → 시각적 특징 의존
- 가려진 관절 (low confidence) → HMD 정보 의존

## HMD 정보 구조

3D GT 좌표에서 추출된 9차원 벡터:

```python
hmd_info = [
    right_hand_local,   # (3,) 오른손 로컬 좌표
    left_hand_local,    # (3,) 왼손 로컬 좌표
    hand_distance,      # (1,) 양손 간 거리
    right_distance,     # (1,) 머리-오른손 거리
    left_distance       # (1,) 머리-왼손 거리
]
```

로컬 좌표계: 머리(Spine2) 기준, Z축은 머리→양손중점 방향

## 훈련 방법

### Conda 환경 설정 (중요!)

훈련 전 반드시 올바른 conda 환경을 활성화해야 합니다.

**문제 상황:**
```bash
# dist_train.sh 실행 시 다음 에러 발생
ModuleNotFoundError: No module named 'torch'
```

**원인:** `dist_train.sh`가 시스템 기본 Python을 사용하여 torch가 설치되지 않은 환경에서 실행됨

**해결 방법:**

```bash
# 1. Conda 환경 활성화 후 훈련 실행
conda activate mmpose  # 또는 해당 환경 이름

# 2. Single GPU 훈련
python tools/train.py <config_path>

# 3. Multi-GPU 훈련 (2 GPU 예시)
bash tools/dist_train.sh <config_path> 2

# 4. Background 실행 (로그 저장)
nohup bash tools/dist_train.sh <config_path> 2 > /tmp/train.log 2>&1 &
```

**환경 확인:**
```bash
# 현재 conda 환경 확인
conda info --envs

# torch 설치 확인
python -c "import torch; print(torch.__version__)"

# CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 훈련 명령어 예시

```bash
# Baseline
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_config.py

# Augmented + Deeper Decoder
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_augmented_config.py

# Confidence-Weighted HMD Fusion
python tools/train.py my_code/custom_config/HMD_xregopose_confidence_weighted_config.py

# Multi-GPU 훈련 (2 GPU)
bash tools/dist_train.sh my_code/custom_config/HMD_xregopose_h5cache_config.py 2
```

## 평가

```bash
python tools/test.py <config_path> <checkpoint_path>
```

평가 지표: MPJPE (Mean Per Joint Position Error)


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
| **_small** | Smoke test config (1 epoch, no checkpoint) |

---

## Smoke Test 규칙

**목적**: 새 구현체의 훈련 파이프라인이 정상 동작하는지 빠르게 확인

**Config 설정**:
```python
max_epochs = 1              # 1 epoch만
val_interval = 1            # validation 1회
checkpoint = None           # 가중치 저장 안함
visualization = False       # 시각화 비활성화
dataset = small cache       # train_small_1k.h5 / val_small_500.h5
```


full cache dataset 
    cache_file_train = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'


small cache dataset
	cache_file_train = '/mnt/dataset_vol/h5cache/train_small_1k.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/val_small_500.h5'
	


**확인 사항**:
- [ ] Forward pass 정상 동작
- [ ] Loss 계산 정상
- [ ] Backward pass / gradient 흐름
- [ ] Validation loop 정상 동작

**주의**: Smoke test 결과(MPJPE)는 성능 평가에 사용하지 않음

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


