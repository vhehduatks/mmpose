# xR-EgoPose Dataset Structure

> xR-EgoPose: Egocentric 3D Human Pose from an HMD Camera
> Tome et al., IEEE TPAMI 2019
> https://github.com/facebookresearch/xR-EgoPose

## Overview

xR-EgoPose is an egocentric pose estimation dataset captured from a head-mounted display (HMD) camera. It provides RGB images with 2D/3D joint annotations for training egocentric pose estimation models.

**H5 Cache Location**: `/mnt/dataset_vol/h5cache/`

---

## Dataset Size

| Split | Samples | File | Size |
|-------|---------|------|------|
| **Training** | 210,454 | `train_cache_with_images.h5` | ~41 GB |
| **Validation** | ~15,000 | `val_cache_with_images.h5` | ~3 GB |
| **Test** | ~115,000 | `test_cache_with_images.h5` | ~16 GB |
| **Small (train)** | 1,000 | `train_small_1k.h5` | ~197 MB |
| **Small (val)** | 500 | `val_small_500.h5` | ~99 MB |

---

## H5 Cache Structure (v2.0)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `images` | (N, 256, 256, 3) | uint8 | RGB images (HWC format) |
| `keypoints` | (N, 1, 16, 2) | float32 | 2D keypoints (pixel coordinates) |
| `keypoint3d` | (N, 1, 16, 3) | float32 | 3D keypoints (meters, camera coords) |
| `hmd_info` | (N, 1, 9) | float32 | HMD preprocessed features |
| `img_paths` | (N,) | string | Original image paths |
| `actions` | (N,) | string | Action labels |

**Attributes:**
- `version`: '2.0'
- `has_images`: True
- `img_size`: 256
- `n_samples`: 210454

---

## Joint Definition (16 Joints)

> Based on `mmpose/datasets/datasets/body3d/egopose_info.py`

```
                    [0] Spine2 (root)
                    /   |   \
           [2] L.Arm  [1] Head  [5] R.Arm
               |                    |
        [3] L.ForeArm          [6] R.ForeArm
               |                    |
          [4] L.Hand            [7] R.Hand

                    [0] Spine2
                    /         \
          [8] L.UpLeg        [12] R.UpLeg
               |                  |
           [9] L.Leg          [13] R.Leg
               |                  |
          [10] L.Foot         [14] R.Foot
               |                  |
         [11] L.ToeBase       [15] R.ToeBase
```

### Joint Index Table

| Index | Joint Name | Type | Swap |
|-------|------------|------|------|
| 0 | Spine2 | Upper (root) | - |
| 1 | Head | Upper | - |
| 2 | LeftArm | Upper | RightArm (5) |
| 3 | LeftForeArm | Upper | RightForeArm (6) |
| 4 | LeftHand | Upper | RightHand (7) |
| 5 | RightArm | Upper | LeftArm (2) |
| 6 | RightForeArm | Upper | LeftForeArm (3) |
| 7 | RightHand | Upper | LeftHand (4) |
| 8 | LeftUpLeg | Lower | RightUpLeg (12) |
| 9 | LeftLeg | Lower | RightLeg (13) |
| 10 | LeftFoot | Lower | RightFoot (14) |
| 11 | LeftToeBase | Lower | RightToeBase (15) |
| 12 | RightUpLeg | Lower | LeftUpLeg (8) |
| 13 | RightLeg | Lower | LeftLeg (9) |
| 14 | RightFoot | Lower | LeftFoot (10) |
| 15 | RightToeBase | Lower | LeftToeBase (11) |

### Skeleton Connections (15 bones)

```python
EGOPOSE_SKELETON = [
    (0, 1),    # Spine2 -> Head
    (0, 2),    # Spine2 -> LeftArm
    (2, 3),    # LeftArm -> LeftForeArm
    (3, 4),    # LeftForeArm -> LeftHand
    (0, 5),    # Spine2 -> RightArm
    (5, 6),    # RightArm -> RightForeArm
    (6, 7),    # RightForeArm -> RightHand
    (0, 8),    # Spine2 -> LeftUpLeg
    (8, 9),    # LeftUpLeg -> LeftLeg
    (9, 10),   # LeftLeg -> LeftFoot
    (10, 11),  # LeftFoot -> LeftToeBase
    (0, 12),   # Spine2 -> RightUpLeg
    (12, 13),  # RightUpLeg -> RightLeg
    (13, 14),  # RightLeg -> RightFoot
    (14, 15),  # RightFoot -> RightToeBase
]
```

---

## Coordinate System

### ⚠️ IMPORTANT: Axis Convention

The xR-EgoPose dataset uses a **camera-centric coordinate system**:

| Axis | Direction | Range (typical) | Description |
|------|-----------|-----------------|-------------|
| **X** | Left/Right | -0.6 ~ +0.6 m | Horizontal |
| **Y** | Forward/Backward | -0.3 ~ +0.1 m | Depth from camera plane |
| **Z** | **Height (Down)** | 0.0 ~ 1.5 m | **Vertical (Z↑ = lower)** |

**Key observations from data analysis:**
```
Sample analysis (Head vs Feet):
  Head:      Z ≈ 0.10 m (low Z = higher position)
  Feet:      Z ≈ 1.40 m (high Z = lower position)
  Difference: ~1.30 m (body height)
```

### Root Position (Spine2)

The root joint (Spine2, index 0) is **NOT at origin**. It has a constant position:
```python
Spine2 = [-0.00668, -0.18734, 0.0467]  # Same for all samples
```

This means the data is in a **global camera coordinate system**, not root-relative.

---

## ✅ Ground Info Bug FIXED (2026-02-03)

The `EnhanceHMDInfo` transform was using **Y-axis** for ground calculation, which is WRONG in camera coordinates.

### Problem
In camera coordinates, **no axis corresponds to "up/down"**. The camera orientation changes as the user moves their head.

### Solution: Body-Axis Method (Vector A)
Instead of using coordinate axes, we use the **body's skeletal structure** to define "downward":

```python
# Vector A = direction from root (Spine2) toward pelvis center
pelvis_center = (left_upleg + right_upleg) / 2
vec_a = pelvis_center - root
vec_a_unit = vec_a / np.linalg.norm(vec_a)

# Ground = farthest toe projection along Vector A
left_toe_proj = np.dot(left_toe - root, vec_a_unit)
right_toe_proj = np.dot(right_toe - root, vec_a_unit)
ground_ref_proj = max(left_toe_proj, right_toe_proj)

# Heights from ground (along body axis)
root_from_ground = ground_ref_proj
hand_from_ground = ground_ref_proj - np.dot(hand - root, vec_a_unit)
```

### Comparison

| Method | Value | Interpretation |
|--------|-------|----------------|
| OLD (Y-axis) | ~0.23m | ❌ WRONG - meaningless |
| NEW (Body-axis) | ~1.43m | ✅ Correct body height |

### Affected Experiments
Experiments #29-35 in EXPERIMENT_RESULTS.md used the buggy Y-axis method and are **INVALID**.
V3 ablation experiments are being re-run with the fixed body-axis method.

---

## HMD Info Structure (9-dim)

The `hmd_info` contains preprocessed head-mounted display sensor data:

| Index | Name | Description |
|-------|------|-------------|
| 0-2 | `right_local` | Right hand position in local coordinate system |
| 3-5 | `left_local` | Left hand position in local coordinate system |
| 6 | `hand_distance` | Distance between both hands |
| 7 | `right_distance` | Head-to-right-hand distance |
| 8 | `left_distance` | Head-to-left-hand distance |

### HMD Info Computation

From `custom_egopose_dataset_h5cache.py`:

```python
def _preprocess_hmd_data(self, p3d):
    head = p3d[0]        # Spine2 (index 0, used as "head" reference)
    right_hand = p3d[7]  # RightHand
    left_hand = p3d[4]   # LeftHand

    # Create local coordinate system from head-hands triangle
    midpoint = (right_hand + left_hand) / 2
    z_axis = normalize(midpoint - head)
    x_axis = normalize(cross(z_axis, right_hand - left_hand))
    y_axis = cross(z_axis, x_axis)
    rotation_matrix = [x_axis, y_axis, z_axis]

    # Transform hands to local coordinates
    right_local = rotation_matrix.T @ (right_hand - head)
    left_local = rotation_matrix.T @ (left_hand - head)

    # Compute distances
    hand_distance = ||right_local - left_local||
    right_distance = ||right_local||
    left_distance = ||left_local||

    return [right_local, left_local, hand_distance, right_distance, left_distance]
```

**Note:** The HMD info uses **Spine2 (index 0)** as the "head" reference, not the actual Head joint (index 1).

---

## Enhanced HMD Info (for Ground Reference)

The `EnhanceHMDInfo` transform adds additional dimensions:

| Mode | Extra Dims | Total | Description |
|------|------------|-------|-------------|
| `head_from_ground` | 1 | 10 | Head height from ground |
| `hand_from_ground` | 2 | 11 | Left/Right hand heights from ground |
| `ground_reference` | 2 | 11 | Head height + head-torso distance |
| `both_from_ground` | 3 | 12 | Head + both hands from ground |

**⚠️ These modes currently have the Y/Z axis bug described above.**

---

## Action Labels

The dataset includes 10 action categories:

| ID | Action | Example animations |
|----|--------|-------------------|
| 0 | Gesticuling | Opening_A_Lid, Angry_Gesture, Waving_Gesture |
| 1 | Reacting | Rejected, Angry_Point, Terrified, Surprised |
| 2 | Greeting | Shaking_Hands_2, Standing_Greeting, Quick_Formal_Bow |
| 3 | Talking | Happy, Plotting, Agreeing, Insult |
| 4 | UpperStretching | Counting, Thinking, Standing_Thumbs_Up |
| 5 | Gaming | Dribble, Boxing, Shooting_Gun, Golf_Putt |
| 6 | LowerStretching | Sitting_Thumbs_Up, Sitting_Disapproval |
| 7 | Patting | Patting, Petting, Petting_Animal |
| 8 | Walking | anim_Clip1, Weight_Shift_Gesture, walking |
| 9 | All | (aggregate) |

---

## Comparison with Mo2Cap2 Dataset

| Feature | xR-EgoPose | Mo2Cap2 |
|---------|------------|---------|
| **Joints** | 16 | 15 |
| **Root Joint** | Spine2 (0) | Neck (0) |
| **Image Type** | Perspective (HMD) | Fisheye (Cap) |
| **Training Samples** | 210,454 | 530,000 |
| **Test Samples** | ~115,000 | ~5,600 |
| **Image Resolution** | 256×256 | 256×256 |
| **Coordinate System** | Camera-centric (Z=height) | Camera-centric |
| **Root Position** | Fixed offset | Fixed offset |
| **HMD Info** | 9-dim | 9-dim (precomputed) |
| **Has ToeBase** | Yes (11, 15) | Yes (10, 14) |
| **Has Head** | Yes (1) | No (Neck only) |

---

## Loading Data (Python)

### Load from H5 Cache

```python
import h5py
import numpy as np

cache_path = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
sample_id = 0

with h5py.File(cache_path, 'r') as f:
    image = f['images'][sample_id]           # (256, 256, 3) uint8
    keypoints = f['keypoints'][sample_id][0]  # (16, 2) float32
    keypoint3d = f['keypoint3d'][sample_id][0]  # (16, 3) float32
    hmd_info = f['hmd_info'][sample_id][0]    # (9,) float32
    action = f['actions'][sample_id]          # string
```

### Compute Ground Info (Correct Implementation)

```python
def compute_ground_info(keypoint3d):
    """Compute ground info using Z-axis (correct for xR-EgoPose)."""
    head = keypoint3d[1]      # Head joint (not Spine2)
    left_foot = keypoint3d[10]
    right_foot = keypoint3d[14]

    # Ground is max Z (lowest point in camera coords where Z↓)
    ground_z = max(left_foot[2], right_foot[2])

    # Heights from ground
    head_from_ground = ground_z - head[2]

    return {
        'ground_z': ground_z,
        'head_from_ground': head_from_ground,
    }
```

---

## References

- **Paper**: [xR-EgoPose: Egocentric 3D Human Pose from an HMD Camera](https://arxiv.org/abs/1907.10045)
- **GitHub**: https://github.com/facebookresearch/xR-EgoPose
- **Dataset Info**: `mmpose/datasets/datasets/body3d/egopose_info.py`
- **H5 Cache Builder**: `tools/dataset_converters/build_egopose_h5cache_with_images.py`
- **Transform (buggy)**: `mmpose/datasets/transforms/enhance_hmd_info.py`
