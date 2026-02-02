# Visualization Scripts

This folder contains scripts for visualizing the XR EgoPose dataset and checking data quality.

## Scripts

### check_skeleton_labeling.py

Visualize skeleton with joint indices and names to verify skeleton labeling is correct.

**Purpose:**
- Verify joint labeling matches expected body parts
- Compare two different skeleton orderings in the codebase
- Debug skeleton visualization issues

**Usage:**

```bash
# Basic usage - visualize samples from H5 cache
python my_code/visualization/check_skeleton_labeling.py \
    --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
    --num-samples 5 \
    --output output_skeleton_check/

# Visualize specific sample index
python my_code/visualization/check_skeleton_labeling.py \
    --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
    --sample-idx 100 \
    --output output_skeleton_check/

# Try egopose_info.py ordering (if visualization looks wrong)
python my_code/visualization/check_skeleton_labeling.py \
    --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
    --use-egopose-info-order \
    --output output_skeleton_check/

# Random sampling
python my_code/visualization/check_skeleton_labeling.py \
    --cache-file /mnt/dataset_vol/h5cache/train_cache_with_images.h5 \
    --num-samples 10 \
    --random \
    --output output_skeleton_check/
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--cache-file` | Path to H5 cache file | Required |
| `--sample-idx` | Specific sample index | None |
| `--num-samples` | Number of samples | 5 |
| `--output` | Output directory | `output_skeleton_check` |
| `--show` | Show visualization window | False |
| `--random` | Random sample selection | False |
| `--use-egopose-info-order` | Use egopose_info.py ordering | False |

**Output:**

For each sample, generates a visualization with:
- 2D skeleton on image with labeled joints
- 3D skeleton from 4 different views (side, front, top, back)
- Joint index reference legend

**Skeleton Ordering Note:**

There are two different skeleton orderings in this codebase:

| Index | config.py (H5 data) | egopose_info.py (vis) |
|-------|---------------------|----------------------|
| 0 | Head | Spine2 |
| 1 | Neck | Head |
| 2-15 | Same | Same |

By default, this script uses `config.py` ordering which matches the H5 cache data.

---

### visualize_dataset.py

Visualize dataset samples including images, keypoints, heatmaps, and HMD info.

**Purpose:**
- Debug data pipeline issues
- Verify data preprocessing is correct
- Inspect heatmap generation

**Usage:**

```bash
# Visualize training samples
python my_code/visualization/visualize_dataset.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --split train \
    --num-samples 5 \
    --output output_dataset_vis/

# Visualize validation samples
python my_code/visualization/visualize_dataset.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --split val \
    --num-samples 10 \
    --output output_dataset_vis/

# Random sampling with debug info
python my_code/visualization/visualize_dataset.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --random \
    --debug \
    --num-samples 5 \
    --output output_dataset_vis/
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config file | Required |
| `--split` | Dataset split (train/val/test) | `train` |
| `--num-samples` | Number of samples | 5 |
| `--output` | Output directory | `output_dataset_vis` |
| `--show` | Show visualization window | False |
| `--random` | Random sample selection | False |
| `--debug` | Print debug info | False |

**Output:**

For each sample, generates two visualization files:

1. **Main visualization** (`{split}_sample_{idx:05d}.png`):
   - Input image (preprocessed)
   - 2D keypoints overlaid on image
   - 3D keypoints visualization
   - HMD direction vectors
   - Individual heatmaps for each joint (16 joints)

2. **3D skeleton detail** (`{split}_sample_{idx:05d}_3d_skeleton.png`):
   - Enlarged 3D skeleton with joint labels
   - 4 views: Perspective, Front, Side, Top
   - Each joint labeled with index and name (e.g., "0:Spine2")
   - **HMD-tracked joints highlighted**:
     - Head (1), LeftHand (4), RightHand (7)
     - Star marker (★) with red edge
     - Red label text with [HMD] suffix

---

## Color Coding Convention

Both scripts use consistent color coding:

| Body Part | Color |
|-----------|-------|
| Head/Neck | Blue (#3399FF) |
| Left Arm | Green (#00FF00) |
| Right Arm | Orange (#FF8000) |
| Left Leg | Cyan (#00FFFF) |
| Right Leg | Magenta (#FF00FF) |

## Joint Index Reference (config.py order)

```
 0: Head           8: LeftUpLeg (Hip)
 1: Neck           9: LeftLeg (Knee)
 2: LeftArm       10: LeftFoot (Ankle)
 3: LeftForeArm   11: LeftToeBase
 4: LeftHand      12: RightUpLeg (Hip)
 5: RightArm      13: RightLeg (Knee)
 6: RightForeArm  14: RightFoot (Ankle)
 7: RightHand     15: RightToeBase
```
