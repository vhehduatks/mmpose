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

```bash
# Baseline
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_config.py

# Augmented + Deeper Decoder
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_augmented_config.py

# Confidence-Weighted HMD Fusion
python tools/train.py my_code/custom_config/HMD_xregopose_confidence_weighted_config.py
```

## 평가

```bash
python tools/test.py <config_path> <checkpoint_path>
```

평가 지표: MPJPE (Mean Per Joint Position Error)
