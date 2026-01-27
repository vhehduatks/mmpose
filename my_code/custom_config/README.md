# XR EgoPose Custom Configs

Custom configuration files for egocentric 3D pose estimation.

## Model Comparison

| Config | Head Type | Features |
|--------|-----------|------|
| `HMD_xregopose_h5cache_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | Baseline (H5 cache) |
| `HMD_xregopose_h5cache_augmented_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | + Data Augmentation + Deeper Decoder |
| `HMD_xregopose_confidence_weighted_config.py` | `ConfidenceWeightedHMDHead` | + Confidence-weighted HMD Fusion |

## Model Improvements

### 1. Data Augmentation (augmented_config)

Only color-based augmentation applied (maintaining 3D coordinate compatibility):

```python
dict(
    type='PhotometricDistortion',
    brightness_delta=32,
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_delta=15,
)
```

> **Note**: RandomFlip, RandomBBoxTransform are excluded as they require 3D GT coordinate transformation

### 2. Pose Decoder Depth (augmented_config)

Increased number of residual blocks:

| Setting | num_stage | Parameters |
|------|-----------|------------|
| Original | 1 | 586K |
| Improved | 2 | 1.1M |

```python
head=dict(
    pose_decoder_num_stage=2,
    pose_decoder_linear_size=512,
    pose_decoder_dropout=0.3,
)
```

### 3. Confidence-Weighted HMD Fusion (confidence_weighted_config)

Dynamic weighting based on heatmap confidence:

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

**How it works:**
```python
# Extract per-joint confidence from heatmap
confidence = sigmoid(heatmap.max(dim=(-1,-2)))  # [B, 16]

# Confidence → weights (learnable projection)
visual_weight = confidence_proj(confidence)     # [B, 64]
hmd_weight = 1 - visual_weight

# Weighted fusion
z_fused = z_visual * visual_weight + z_hmd * hmd_weight
```

**Expected effect:**
- Visible joints (high confidence) → relies on visual features
- Occluded joints (low confidence) → relies on HMD information

## HMD Information Structure

9-dimensional vector extracted from 3D GT coordinates:

```python
hmd_info = [
    right_hand_local,   # (3,) right hand local coordinates
    left_hand_local,    # (3,) left hand local coordinates
    hand_distance,      # (1,) distance between both hands
    right_distance,     # (1,) head-to-right-hand distance
    left_distance       # (1,) head-to-left-hand distance
]
```

Local coordinate system: relative to head (Spine2), Z-axis points from head to midpoint of both hands

## Training

### Conda Environment Setup (Important!)

You must activate the correct conda environment before training.

**Problem scenario:**
```bash
# The following error occurs when running dist_train.sh
ModuleNotFoundError: No module named 'torch'
```

**Cause:** `dist_train.sh` uses the system default Python, running in an environment where torch is not installed

**Solution:**

```bash
# 1. Activate conda environment then run training
conda activate mmpose  # or the appropriate environment name

# 2. Single GPU training
python tools/train.py <config_path>

# 3. Multi-GPU training (2 GPU example)
bash tools/dist_train.sh <config_path> 2

# 4. Background execution (save logs)
nohup bash tools/dist_train.sh <config_path> 2 > /tmp/train.log 2>&1 &
```

**Environment verification:**
```bash
# Check current conda environment
conda info --envs

# Check torch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Training Command Examples

```bash
# Baseline
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_config.py

# Augmented + Deeper Decoder
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_augmented_config.py

# Confidence-Weighted HMD Fusion
python tools/train.py my_code/custom_config/HMD_xregopose_confidence_weighted_config.py

# Multi-GPU training (2 GPU)
bash tools/dist_train.sh my_code/custom_config/HMD_xregopose_h5cache_config.py 2
```

## Evaluation

```bash
python tools/test.py <config_path> <checkpoint_path>
```

Evaluation metric: MPJPE (Mean Per Joint Position Error)


---

## Experiment Order and Schedule

| Order | Experiment | Estimated Duration | Dependencies |
|------|------|----------|--------|
| 1 | Phase 1: Progressive Warmup | 1 day | None (implementation complete) |
| 2 | Baseline: Dual COCO+MPII (original) | 0.5 days | Parallel with Phase 1 |
| 3 | Phase 2-A: Warmup + Ensemble | 1 day | After Phase 1 completion |
| 4 | Phase 3-A: Warmup + KL Div | 1 day | After Phase 1 completion |
| 5 | Phase 3-B: Full combination | 1 day | After Phase 2, 3 results review |
| 6 | Phase 5-A: Decoder optimization | 2 days | After optimal combination finalized |

---

## Evaluation Metrics

### Primary Metrics
- **Full Body MPJPE** (mm): Main evaluation metric
- **Upper Body MPJPE** (mm)
- **Lower Body MPJPE** (mm)

### Auxiliary Metrics
- `mutual_weight`: Progressive warmup progress status
- `loss_backbone_latant`: Mutual learning loss value
- `acc_pose`: 2D heatmap accuracy

### Monitoring (Wandb)
- Loss curves: Trend per loss component
- Learning rate schedule
- GPU memory usage

---

## Config File Naming Convention

```
HMD_xregopose_h5cache_coco_mpii_{variant}_config.py
```

| Variant | Description |
|---------|------|
| (none) | Original dual backbone |
| warmup | Phase 1: Progressive warmup |
| ensemble | Phase 2: Ensemble teacher |
| kldiv | Phase 3: KL divergence |
| full | Full combination |
| lifting | Phase 5-B: 2D→3D lifting |
| **_small** | Smoke test config (1 epoch, no checkpoint) |

---

## Smoke Test Rules

**Purpose**: Quickly verify that the training pipeline of a new implementation works correctly

**Config settings**:
```python
max_epochs = 1              # 1 epoch only
val_interval = 1            # 1 validation run
checkpoint = None           # No weight saving
visualization = False       # Disable visualization
dataset = small cache       # train_small_1k.h5 / val_small_500.h5
```


full cache dataset
    cache_file_train = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'


small cache dataset
	cache_file_train = '/mnt/dataset_vol/h5cache/train_small_1k.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/val_small_500.h5'



**Checklist**:
- [ ] Forward pass works correctly
- [ ] Loss computation is correct
- [ ] Backward pass / gradient flow
- [ ] Validation loop works correctly

**Note**: Smoke test results (MPJPE) should not be used for performance evaluation

---

## Result Recording Template

### Experiment: Phase X - [Experiment Name]

**Config**: `HMD_xregopose_h5cache_coco_mpii_xxx_config.py`

**Hyperparameters**:
| Parameter | Value |
|---------|-----|
| mutual_warmup_epochs | |
| mutual_rampup_epochs | |
| temperature | |
| ... | |

**Results**:
| Metric | Value |
|--------|-------|
| Full Body MPJPE | mm |
| Upper Body MPJPE | mm |
| Lower Body MPJPE | mm |
| Best Epoch | |

**Observations**:
-

**Conclusions**:
-

---

