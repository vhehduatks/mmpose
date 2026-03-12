# Mo2Cap2 V2 Cache X Coordinate Offset (-33)

## Overview

Mo2Cap2 dataset requires a **-33 pixel X coordinate offset** for proper 2D keypoint alignment with images. This document explains how the offset is applied in different components.

## Data Sources

| Data Source | File | X Offset Applied? |
|-------------|------|-------------------|
| Raw H5 Chunks | `mo2cap2_chunk_*.hdf5` | ❌ No |
| Annotation Cache | `annotations_cache.h5` | ✅ Yes |

## X Coordinate Values (Sample 0)

| Source | X Range |
|--------|---------|
| Raw H5 Chunk | 143.1 ~ 209.8 |
| Annotation Cache | 110.1 ~ 176.8 |
| Difference | **-33** |

## Data Flow

```
Raw H5 Chunk (Annot2D)          Annotation Cache (keypoints)
X: 143.1 ~ 209.8                X: 110.1 ~ 176.8
        │                               │
        │ apply -33                     │ (already applied when cache was built)
        ▼                               ▼
Final: 110.1 ~ 176.8            Final: 110.1 ~ 176.8
     (same result)                   (same result)
```

## Component Behavior

| Component | Data Source | Action |
|-----------|-------------|--------|
| **Training Pipeline** (`H5Mo2Cap2Dataset`) | Annotation cache | No additional offset (already in cache) |
| **Visualization Script** (`visualize_mo2cap2_v2_cache.py`) | Raw H5 chunks | Must apply -33 offset |

## Code References

### Training Pipeline (h5_mo2cap2_dataset.py)

**When building cache** (applies -33 before saving):
```python
# Line 278-283
annot2d = hf['Annot2D'][:]  # Load raw
annot2d[:, :, 0] = annot2d[:, :, 0] - 33  # Apply offset
# ... then save to cache
```

**When loading cache** (no additional offset):
```python
# Line 253-264
with h5py.File(cache_path, 'r') as hf:
    all_keypoints = hf['keypoints'][:]  # Already has -33 applied
# NOTE: X offset (-33) is already applied when building the cache
# Do NOT apply it again here
```

### Visualization Script (visualize_mo2cap2_v2_cache.py)

**Reads from raw H5 chunks** (applies -33):
```python
keypoint2d = hf['Annot2D'][local_idx].copy()
# Apply X coordinate offset (-33) for Mo2Cap2 image alignment
keypoint2d[:, 0] = keypoint2d[:, 0] - 33
```

## Bug Fix History (2026-02-05)

**Bug**: The offset was being applied twice in the training pipeline:
1. When building cache: -33 applied ✓
2. When loading cache: -33 applied again ✗ (BUG)

**Result**: Total offset was -66 instead of -33

**Fix**: Removed duplicate offset application when loading from cache.

| Stage | Before Fix | After Fix |
|-------|------------|-----------|
| Raw H5 | X: 143.1 ~ 209.8 | X: 143.1 ~ 209.8 |
| Cache | X: 110.1 ~ 176.8 | X: 110.1 ~ 176.8 |
| Training Pipeline | X: 77.1 ~ 143.8 ❌ | X: 110.1 ~ 176.8 ✓ |

## Why -33 Offset?

From `MO2CAP2_DATASET.md`:

> The X coordinate is offset by -33 pixels to correct for image alignment.

This offset corrects for the image cropping/processing applied when the original dataset was created.

## Related Files

| File | Purpose |
|------|---------|
| `mmpose/datasets/datasets/body3d/h5_mo2cap2_dataset.py` | Dataset class (uses cache) |
| `my_code/visualization/visualize_mo2cap2_v2_cache.py` | Visualization (uses raw H5) |
| `my_code/custom_config/MO2CAP2_DATASET.md` | Dataset documentation |
