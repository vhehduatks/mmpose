# Mo2Cap2 Dataset Structure

> Mo2Cap2: Real-time Mobile 3D Motion Capture with a Cap-mounted Fisheye Camera
> Xu et al., IEEE TVCG Proc. VR 2019

## Overview

Mo2Cap2 is an egocentric pose estimation dataset captured using a **fisheye camera** mounted on a cap. It provides RGB images with 2D/3D joint annotations for training egocentric pose estimation models.

**Dataset Location**: `/mnt/sdb2/mo2cap2_dataset`

---

## Directory Structure

```
/mnt/sdb2/mo2cap2_dataset/
├── README.txt                          # Dataset description & citation
├── license.txt                         # Usage license (research only)
├── download.sh                         # Download script
├── conf.ig                             # Configuration
│
├── training_data/                      # Training set (530k samples)
│   ├── annotations_cache.h5            # Precomputed annotation cache (150MB)
│   ├── mo2cap2_chunk_0001.hdf5         # Chunk 1 (1000 samples, ~405MB)
│   ├── mo2cap2_chunk_0002.hdf5         # Chunk 2
│   └── ... (530 chunks total)
│
├── test_data/                          # Test set (5.6k samples)
│   └── TestSet/
│       ├── olek_outdoor/               # Outdoor sequence (2,744 JPG images)
│       ├── olek_outdoor_gt.mat         # Ground truth poses
│       ├── weipeng_studio/             # Studio sequence (2,902 JPG images)
│       └── weipeng_studio_gt.mat       # Ground truth poses
│
└── code/                               # Evaluation code (MATLAB)
```

---

## Dataset Size

| Split | Samples | Format | Size |
|-------|---------|--------|------|
| **Training** | 530,000 | 530 HDF5 chunks × 1000 | ~210 GB |
| **Test (olek_outdoor)** | 2,744 | JPG + MAT | ~3 GB |
| **Test (weipeng_studio)** | 2,902 | JPG + MAT | ~3 GB |
| **Total** | 535,646 | - | ~216 GB |

---

## Training Data Structure

### HDF5 Chunk Structure (per file)

Each `mo2cap2_chunk_XXXX.hdf5` contains 1000 samples:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `Images` | (1000, 3, 256, 256) | uint8 | Original fisheye RGB images (CHW format) |
| `ZoomImages` | (1000, 3, 256, 256) | uint8 | Zoomed/cropped images |
| `Annot2D` | (1000, 15, 2) | float64 | 2D joint coordinates (pixel) |
| `Annot3D` | (1000, 15, 3) | float64 | 3D joint coordinates (meters) |
| `Heatmaps` | (1000, 15, 32, 32) | uint8 | Joint heatmaps (low-res) |
| `ZoomHeatmaps` | (1000, 15, 32, 32) | uint8 | Zoomed heatmaps |

### Annotations Cache

`annotations_cache.h5` provides quick access to all annotations:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `keypoint3d` | (530000, 15, 3) | float32 | All 3D keypoints |
| `keypoints` | (530000, 15, 2) | float32 | All 2D keypoints |
| `hmd_info` | (530000, 9) | float32 | HMD info (precomputed) |

---

## Test Data Structure

### Image Format
- **Format**: JPEG
- **Naming**: `fc2_save_YYYY-MM-DD-HHMMSS-XXXX.jpg`
- **Resolution**: Variable (fisheye)

### Ground Truth (MAT file)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `pose_gt` | (N, 15, 3) | float64 | 3D ground truth poses |

**Note**: GT may have more frames than images due to frame synchronization.

---

## Joint Definition (15 Joints)

> **Note**: Based on `configs/_base_/datasets/custom_mo2cap2.py`. The skeleton has no explicit Head or Hip joints - the Neck serves as the upper root, and arms connect directly to upper legs (implicit torso).

```
                    [0] Neck
                    /       \
          [4] L.Arm           [1] R.Arm
              |       \   /       |
       [5] L.ForeArm   \ /   [2] R.ForeArm
              |         X         |
          [6] L.Hand   / \    [3] R.Hand
                      /   \
          [11] L.UpLeg     [7] R.UpLeg
              |               |
         [12] L.Leg        [8] R.Leg
              |               |
         [13] L.Foot       [9] R.Foot
              |               |
        [14] L.ToeBase    [10] R.ToeBase
```

### Joint Index Table

| Index | Joint Name | Side | Type |
|-------|------------|------|------|
| 0 | Neck | Center | Upper |
| 1 | RightArm | Right | Upper |
| 2 | RightForeArm | Right | Upper |
| 3 | RightHand | Right | Upper |
| 4 | LeftArm | Left | Upper |
| 5 | LeftForeArm | Left | Upper |
| 6 | LeftHand | Left | Upper |
| 7 | RightUpLeg | Right | Lower |
| 8 | RightLeg | Right | Lower |
| 9 | RightFoot | Right | Lower |
| 10 | RightToeBase | Right | Lower |
| 11 | LeftUpLeg | Left | Lower |
| 12 | LeftLeg | Left | Lower |
| 13 | LeftFoot | Left | Lower |
| 14 | LeftToeBase | Left | Lower |

### Skeleton Connections (14 bones)

```python
# Based on configs/_base_/datasets/custom_mo2cap2.py skeleton_info
MO2CAP2_SKELETON = [
    (0, 4),    # Neck -> LeftArm
    (0, 1),    # Neck -> RightArm
    (4, 5),    # LeftArm -> LeftForeArm
    (5, 6),    # LeftForeArm -> LeftHand
    (1, 2),    # RightArm -> RightForeArm
    (2, 3),    # RightForeArm -> RightHand
    (4, 11),   # LeftArm -> LeftUpLeg (torso connection)
    (11, 12),  # LeftUpLeg -> LeftLeg
    (12, 13),  # LeftLeg -> LeftFoot
    (13, 14),  # LeftFoot -> LeftToeBase
    (1, 7),    # RightArm -> RightUpLeg (torso connection)
    (7, 8),    # RightUpLeg -> RightLeg
    (8, 9),    # RightLeg -> RightFoot
    (9, 10),   # RightFoot -> RightToeBase
]
```

**Special Notes**:
- No explicit Hip/Torso joint - arms connect directly to upper legs (LeftArm→LeftUpLeg, RightArm→RightUpLeg)
- No Head joint - only Neck (index 0)
- Includes ToeBase joints for detailed foot tracking

---

## Preprocessing (from mo2cap2_coco_dataset.py)

When loading Mo2Cap2 data, the following preprocessing is applied:

### 1. 2D Coordinate Offset
```python
p2d[jid][0] = frame_data[joint_name]['2d'][0] - 33  # X coordinate only
p2d[jid][1] = frame_data[joint_name]['2d'][1]       # Y coordinate unchanged
```
The X coordinate is offset by -33 pixels to correct for image alignment.

### 2. 3D Coordinate Normalization (Neck as Reference)
```python
p3d -= p3d[0]  # Set Neck (index 0) as origin (0,0,0)
```
All 3D coordinates are normalized so that Neck becomes the reference point at (0,0,0).

### 3. Unit Conversion
```python
p3d /= 1000  # mm → meters
hmd_info /= 1000
```

---

## HMD Info Structure (9-dim)

The `hmd_info` in annotations_cache contains head-mounted display sensor data:

| Index | Name | Description |
|-------|------|-------------|
| 0-2 | `right_hand_local` | Right hand position relative to head (x, y, z) |
| 3-5 | `left_hand_local` | Left hand position relative to head (x, y, z) |
| 6 | `hand_distance` | Distance between both hands |
| 7 | `right_distance` | Head-to-right-hand distance |
| 8 | `left_distance` | Head-to-left-hand distance |

---

## Comparison with EgoPose Dataset

| Feature | Mo2Cap2 | EgoPose |
|---------|---------|---------|
| **Joints** | 15 | 16 |
| **Root Joint** | Neck (0) | Spine2 (0) |
| **Image Type** | Fisheye | Perspective |
| **Training Samples** | 530,000 | 210,000 |
| **Test Samples** | 5,646 | 115,000 |
| **Image Resolution** | 256×256 | 256×256 |
| **Heatmap Resolution** | 32×32 | 47×47 |
| **Data Format** | HDF5 chunks | H5 cache |
| **HMD Info** | 9-dim (precomputed) | 9-dim |

### Joint Mapping (Mo2Cap2 → EgoPose)

| Mo2Cap2 | EgoPose | Notes |
|---------|---------|-------|
| Neck (0) | Spine2 (0) | Root joint (different semantics) |
| RightArm (1) | RightArm (5) | Index differs |
| RightForeArm (2) | RightForeArm (6) | Index differs |
| RightHand (3) | RightHand (7) | Index differs |
| LeftArm (4) | LeftArm (2) | Index differs |
| LeftForeArm (5) | LeftForeArm (3) | Index differs |
| LeftHand (6) | LeftHand (4) | Index differs |
| RightUpLeg (7) | RightUpLeg (12) | Index differs |
| RightLeg (8) | RightLeg (13) | Index differs |
| RightFoot (9) | RightFoot (14) | Index differs |
| RightToeBase (10) | RightToeBase (15) | Mo2Cap2 has it! |
| LeftUpLeg (11) | LeftUpLeg (8) | Index differs |
| LeftLeg (12) | LeftLeg (9) | Index differs |
| LeftFoot (13) | LeftFoot (10) | Index differs |
| LeftToeBase (14) | LeftToeBase (11) | Mo2Cap2 has it! |
| - | Head (1) | Mo2Cap2 missing |

---

## Loading Data (Python)

### Load Training Sample

```python
import h5py
import numpy as np

chunk_id = 1
sample_id = 0

with h5py.File(f'/mnt/sdb2/mo2cap2_dataset/training_data/mo2cap2_chunk_{chunk_id:04d}.hdf5', 'r') as f:
    # Images are (N, C, H, W), transpose to (H, W, C) for visualization
    image = f['Images'][sample_id].transpose(1, 2, 0)
    annot_2d = f['Annot2D'][sample_id]  # (15, 2)
    annot_3d = f['Annot3D'][sample_id]  # (15, 3)
    heatmaps = f['Heatmaps'][sample_id]  # (15, 32, 32)
```

### Load Annotations Cache

```python
with h5py.File('/mnt/sdb2/mo2cap2_dataset/training_data/annotations_cache.h5', 'r') as f:
    keypoint3d = f['keypoint3d'][:]  # (530000, 15, 3)
    keypoints = f['keypoints'][:]    # (530000, 15, 2)
    hmd_info = f['hmd_info'][:]      # (530000, 9)
```

### Load Test Sample

```python
import scipy.io as sio
from PIL import Image
import os

test_set = 'olek_outdoor'
test_dir = f'/mnt/sdb2/mo2cap2_dataset/test_data/TestSet/{test_set}'

# Load image
images = sorted(os.listdir(test_dir))
img = np.array(Image.open(os.path.join(test_dir, images[0])))

# Load ground truth
gt = sio.loadmat(f'/mnt/sdb2/mo2cap2_dataset/test_data/TestSet/{test_set}_gt.mat')
pose_gt = gt['pose_gt']  # (N, 15, 3)
```

---

## Visualization

```bash
# Visualize training sample
python my_code/visualization/visualize_mo2cap2.py --chunk 1 --sample 0

# Visualize test sample
python my_code/visualization/visualize_mo2cap2.py --test olek_outdoor --sample 0

# Save visualization
python my_code/visualization/visualize_mo2cap2.py --chunk 1 --sample 0 --save

# Show annotation statistics
python my_code/visualization/visualize_mo2cap2.py --stats
```

---

## Citation

```bibtex
@article{xu2019mo2cap2,
  title={Mo2Cap2: Real-time Mobile 3D Motion Capture with a Cap-mounted Fisheye Camera},
  author={Xu, Weipeng and Chatterjee, Avishek and Zollh{\"o}fer, Michael and Rhodin, Helge
          and Fua, Pascal and Seidel, Hans-Peter and Theobalt, Christian},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2019},
  publisher={IEEE}
}
```

---

---

## MMPose Integration

The Mo2Cap2 dataset is integrated into MMPose with the following components:

### Key Files

| File | Purpose |
|------|---------|
| `mmpose/datasets/datasets/body3d/h5_mo2cap2_dataset.py` | Dataset class for H5 loading |
| `mmpose/datasets/transforms/loading.py` | `LoadImageFromH5` transform |
| `mmpose/codecs/custom_mo2cap2_msra_heatmap.py` | Keypoint codec |
| `mmpose/evaluation/metrics/custom_mo2cap2_metric.py` | Evaluation metric |
| `mmpose/models/heads/heatmap_heads/custom_mo2cap2_baselinel1_head.py` | Model head |
| `my_code/custom_config/HMD_mo2cap2_h5_config.py` | Training config |

### Dataset Class: `H5Mo2Cap2Dataset`

```python
@DATASETS.register_module(name='H5Mo2Cap2Dataset')
class H5Mo2Cap2Dataset(BaseDataset):
    """Direct HDF5 chunk loading - ~3-5 seconds for 530k samples"""
```

**Key Features**:
1. **Fast Indexing**: Builds chunk index from file listing (~3-5 sec vs ~5-10 min with JSON)
2. **Annotation Cache**: Creates `annotations_cache.h5` with precomputed keypoints and HMD info
3. **Lazy Loading**: Images loaded on-demand via `LoadImageFromH5` transform

### Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  H5Mo2Cap2Dataset.load_data_list()                                          │
│                                                                             │
│  1. _build_chunk_index()                                                    │
│     - List all mo2cap2_chunk_*.hdf5 files                                  │
│     - Build cumulative index for global→local idx mapping                  │
│                                                                             │
│  2. _load_or_build_cache()                                                  │
│     ├─ If annotations_cache.h5 exists → Load directly                      │
│     └─ Else → Build cache:                                                  │
│        a. Load Annot2D, Annot3D from each chunk                            │
│        b. Convert 3D: mm → meters, subtract root (neck)                    │
│        c. Compute HMD info (9-dim) via _preprocess_hmd_data_batch()        │
│        d. Save to annotations_cache.h5                                      │
│                                                                             │
│  3. Create data_info dict for each sample:                                  │
│     - h5_chunk_idx, h5_local_idx, h5_chunk_path                            │
│     - keypoints, keypoint3d, hmd_info (from cache)                         │
│     - bbox, img_id, img_path (virtual)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3D Keypoint Preprocessing

```python
# In _load_or_build_cache():

# 1. Load raw annotations
annot3d = hf['Annot3D'][:]  # (N, 15, 3) in millimeters

# 2. Convert to meters
annot3d_m = annot3d / 1000.0  # MM_TO_M = 1000

# 3. Make root-relative (subtract neck position)
annot3d_m = annot3d_m - annot3d_m[:, 0:1, :]  # Joint 0 = Neck
```

### HMD Info Computation

```python
def _preprocess_hmd_data_batch(self, p3d_batch: np.ndarray) -> np.ndarray:
    """
    Compute 9-dim HMD features from 3D pose.

    Mo2Cap2 joints: 0=Neck(Head), 3=RightHand, 6=LeftHand

    Returns:
        (N, 9) array: [right_local(3), left_local(3),
                       hand_dist(1), right_dist(1), left_dist(1)]
    """
    head = p3d_batch[:, 0]        # (N, 3) - Neck as head
    right_hand = p3d_batch[:, 3]  # (N, 3)
    left_hand = p3d_batch[:, 6]   # (N, 3)

    # Create local coordinate system
    midpoint = (right_hand + left_hand) / 2
    z_axis = normalize(midpoint - head)
    x_axis = normalize(cross(z_axis, right_hand - left_hand))
    y_axis = cross(z_axis, x_axis)

    # Transform to local coordinates
    rotation_matrix = stack([x_axis, y_axis, z_axis])
    right_local = rotation_matrix @ (right_hand - head)
    left_local = rotation_matrix @ (left_hand - head)

    # Compute distances
    hand_distance = norm(right_local - left_local)
    right_distance = norm(right_local)
    left_distance = norm(left_local)

    return concat([right_local, left_local,
                   hand_distance, right_distance, left_distance])
```

### Image Loading Transform

```python
@TRANSFORMS.register_module()
class LoadImageFromH5(object):
    """Load image from HDF5 chunk file."""

    def transform(self, results: dict) -> dict:
        h5_path = results['h5_chunk_path']
        local_idx = results['h5_local_idx']
        use_zoom = results.get('use_zoom', False)

        hf = h5py.File(h5_path, 'r')

        # Images stored as (N, C, H, W) in CHW format
        key = 'ZoomImages' if use_zoom else 'Images'
        img = hf[key][local_idx]  # (3, 256, 256)

        # Convert CHW RGB → HWC BGR (OpenCV format)
        img = img.transpose(1, 2, 0)  # (256, 256, 3)
        img = img[:, :, ::-1]         # RGB → BGR

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results
```

### Training Pipeline

```python
train_pipeline = [
    dict(type='LoadImageFromH5'),           # Load from H5 chunk
    dict(type='GetBBoxCenterScale', padding=1.),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='GenerateTarget', encoder=codec),  # Generate heatmaps
    dict(type='PackPoseInputs')
]
```

### Codec: `Custom_mo2cap2_MSRAHeatmap`

```python
codec = dict(
    type='Custom_mo2cap2_MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(47, 47),  # Upsampled from original 32x32
    sigma=3
)
```

**Key Mappings**:
- `label_mapping_table`: keypoint3d, keypoints, hmd_info → packed into `gt_instance_labels`
- `field_mapping_table`: heatmaps → packed into `gt_fields`

### Config Example

```python
# my_code/custom_config/HMD_mo2cap2_h5_config.py

dataset_mo2cap2_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root='/mnt/sdb2/mo2cap2_dataset/training_data',
    data_mode='topdown',
    pipeline=train_pipeline,
    input_size=(256, 256),
    sample_interval=1,  # Use all 530k samples
    use_zoom=False,     # Use original images (not zoomed)
)
```

---

## Notes

1. **Fisheye Distortion**: Images have significant fisheye distortion, which affects 2D keypoint positions
2. **Root Joint**: Mo2Cap2 uses Neck (0) as root for HMD computation, Hip (14) as skeleton root
3. **Missing Toes**: Mo2Cap2 has no toe joints (15 vs 16 joints)
4. **Heatmap Resolution**: Original 32×32, upsampled to 47×47 in codec
5. **Image Format**: Training uses HDF5 (CHW RGB), converted to HWC BGR in transform
6. **Annotation Cache**: `annotations_cache.h5` created on first run (~150MB)
