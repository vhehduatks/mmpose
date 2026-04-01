# Kinect EgoPose - Comprehensive Experiment Results

## Dataset Overview

| Dataset | Source | Train Sessions | Train Samples | Val/Test Sessions | Val/Test Samples | Data Root |
|---------|--------|---------------|---------------|-------------------|------------------|-----------|
| **V4** | `annotation_egodataset_2026_sei` | 178 | 28,033 | 58 (test) | 11,112 | `/mnt/dataset_vol/annotation_egodataset_2026_sei/` |
| **V3** | `preprocessing_egodataset_weightmodify_intrinsic_ver3` | 32 batches (632 sessions) | - | 9 batches (179 sessions) | - | `/mnt/dataset_vol/kinect_v3_split/` |
| **V3 Blend** | `preprocessing_egodataset_weightmodify_intrinsic_ver3_blend` | 32 batches | - | 9 batches | - | `/mnt/dataset_vol/kinect_v3_blend_split/` |

- V3 and V3 Blend share the same batch-level 8:2 train/val split (same batch IDs)
- Azure Kinect 32 joints → 16 xRegopose joint mapping
- Visibility: `conf_3d >= 2` (Kinect tracking quality), NOT ego-camera visibility

## Common Settings (All Experiments)

| Setting | Value |
|---------|-------|
| Backbone | ResNet-101 (COCO pretrained) |
| Optimizer | AdamW, lr=0.0005 |
| Scheduler | MultiStepLR (gamma=0.5) |
| Batch size | 16 |
| Max epochs | 10 or 20 (see per-experiment) |
| Codec | Custom_mo2cap2_MSRAHeatmap (47x47, sigma=3) |
| Pipeline | LoadImage → EgoImageResize(256x256) → GenerateTarget → PackPoseInputs |
| Image bbox | Full frame (0,0,1920,1080) - no cropping |

---

## Overall Best Results Summary

| Dataset | Model | Epochs | Ground Info (12-dim) | Baseline (9-dim) | No HMD | GI vs Baseline | GI vs No-HMD |
|---------|-------|--------|---------------------|-------------------|--------|----------------|---------------|
| **V4** | Single | 10 | **97.16mm** (ep10) | 102.50mm (ep10) | 107.83mm (ep9) | -5.34mm (-5.2%) | -10.67mm (-9.9%) |
| **V4** | Cascaded | 10 | 98.57mm (ep10) | 106.03mm (ep10) | 107.79mm (ep10) | -7.46mm (-7.0%) | -9.22mm (-8.6%) |
| **V3** | Single | 10 | 72.14mm (ep9) | 75.14mm (ep9) | 76.37mm (ep9) | -3.00mm (-4.0%) | -4.23mm (-5.5%) |
| **V3** | Cascaded | 10 | 72.15mm (ep9) | 74.19mm (ep9) | 79.59mm (ep7) | -2.04mm (-2.7%) | -7.44mm (-9.4%) |
| **V3 Blend** | Single | 10 | 71.52mm (ep9) | 74.72mm (ep9) | 73.98mm (ep9) | -3.20mm (-4.3%) | -2.46mm (-3.3%) |
| **V3 Blend** | Cascaded | 10 | 70.30mm (ep9) | 73.71mm (ep9) | 76.49mm (ep9) | -3.41mm (-4.6%) | -6.19mm (-8.1%) |
| **V3 Blend** | Cascaded | **20** | 72.40mm (ep11) | 73.34mm (ep15) | 76.14mm (ep20) | -0.94mm (-1.3%) | -3.74mm (-4.9%) |
| **V5 Flag** | Single | 10 | 68.77mm (ep10) | 70.69mm (ep8) | 69.41mm (ep9) | -1.92mm (-2.7%) | -0.64mm (-0.9%) |
| **V5 Flag** | Cascaded | 10 | **64.36mm** (ep10) | 70.97mm (ep9) | 71.79mm (ep9) | -6.61mm (-9.3%) | -7.43mm (-10.3%) |

**Overall Best: V5 Flag Cascaded Ground Info (10ep) = 64.36mm** (Upper: 60.91mm, Lower: 67.81mm)

> V5 Flag dataset (`egodataset_flag_fixed_ver5`) uses `use_2d_visible=True` to skip ~2% error frames with unreliable 3D GT. This filtering improved results across the board, with Cascaded Ground Info achieving a new best of 64.36mm (-5.94mm vs V3 Blend best).

---

## V4 Dataset Results

### V4 Single-Stage (CustomxRegoposeBaselinel1)

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 127.06 | 121.12 | 133.00 | 130.57 | 125.41 | 135.74 | 180.70 | 151.28 | 210.12 |
| 2 | 119.42 | 120.57 | 118.27 | 124.36 | 118.87 | 129.85 | 153.50 | 158.12 | 148.88 |
| 3 | 108.03 | 104.53 | 111.53 | 124.40 | 128.67 | 120.12 | 137.91 | 140.19 | 135.64 |
| 4 | 117.81 | 118.50 | 117.11 | 121.54 | 121.76 | 121.32 | 133.61 | 130.08 | 137.13 |
| 5 | 114.53 | 110.89 | 118.17 | 114.54 | 110.53 | 118.56 | 210.56 | 170.26 | 250.85 |
| 6 | 98.15 | 100.08 | 96.23 | 105.62 | 104.43 | 106.80 | 197.14 | 149.10 | 245.19 |
| 7 | 97.35 | 99.03 | 95.66 | 106.20 | 102.32 | 110.08 | 109.45 | 109.65 | 109.25 |
| 8 | 105.99 | 105.27 | 106.71 | 103.58 | 100.37 | 106.80 | 120.94 | 114.77 | 127.10 |
| 9 | 98.05 | 98.64 | 97.47 | 105.95 | 103.50 | 108.40 | **107.83** | **108.44** | **107.22** |
| **10** | **97.16** | **96.73** | **97.60** | **102.50** | **100.10** | **104.90** | 112.02 | 111.81 | 112.23 |

### V4 Cascaded V2b (CustomEgoposeCascadedRefinementHead_enhanced)

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 138.04 | 135.97 | 140.11 | 136.73 | 140.25 | 133.21 | 297.27 | 235.27 | 359.26 |
| 2 | 128.08 | 128.97 | 127.20 | 133.22 | 124.64 | 141.80 | 134.12 | 136.80 | 131.44 |
| 3 | 113.44 | 120.63 | 106.26 | 112.77 | 111.57 | 113.98 | 146.15 | 164.24 | 128.06 |
| 4 | 119.67 | 121.06 | 118.27 | 138.63 | 138.98 | 138.29 | 131.09 | 136.66 | 125.53 |
| 5 | 102.65 | 104.54 | 100.75 | 111.70 | 107.41 | 115.98 | 130.06 | 129.74 | 130.39 |
| 6 | 102.08 | 103.65 | 100.51 | 110.55 | 112.00 | 109.09 | 109.29 | 110.39 | 108.19 |
| 7 | 100.31 | 101.85 | 98.78 | 109.38 | 107.89 | 110.86 | 115.11 | 115.43 | 114.80 |
| 8 | 102.88 | 103.60 | 102.17 | 111.61 | 107.51 | 115.71 | 113.34 | 114.98 | 111.69 |
| 9 | 100.69 | 101.37 | 100.01 | 110.75 | 107.13 | 114.36 | 108.20 | 108.29 | 108.11 |
| **10** | **98.57** | **99.06** | **98.08** | **106.03** | **102.80** | **109.27** | **107.79** | **107.38** | **108.20** |

### V4 Cross-Architecture Comparison

| Model | Single-Stage | Cascaded V2b | Delta |
|-------|-------------|-------------|-------|
| Ground Info | **97.16mm** | 98.57mm | +1.41mm (+1.5%) |
| Baseline | **102.50mm** | 106.03mm | +3.53mm (+3.4%) |
| No-HMD | **107.83mm** | 107.79mm | ~same |

---

## V3 Dataset Results (Original)

### V3 Single-Stage

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 94.26 | 85.49 | 103.03 | 96.17 | 85.66 | 106.69 | 102.41 | 84.05 | 120.78 |
| 2 | 82.90 | 77.87 | 87.94 | 86.26 | 78.66 | 93.86 | 92.04 | 81.13 | 102.96 |
| 3 | 76.85 | 75.06 | 78.65 | 82.66 | 78.02 | 87.29 | 82.82 | 77.34 | 88.30 |
| 4 | 80.16 | 76.42 | 83.91 | 80.92 | 76.68 | 85.16 | 85.36 | 79.97 | 90.74 |
| 5 | 77.08 | 75.08 | 79.08 | 78.87 | 75.11 | 82.63 | 92.91 | 81.50 | 104.33 |
| 6 | 73.28 | 70.11 | 76.44 | 76.57 | 71.77 | 81.37 | 76.89 | 71.62 | 82.15 |
| 7 | 74.58 | 72.56 | 76.60 | 78.27 | 73.35 | 83.19 | 79.15 | 73.93 | 84.37 |
| 8 | 73.49 | 70.99 | 76.00 | 78.14 | 72.76 | 83.51 | 78.64 | 72.80 | 84.48 |
| **9** | **72.14** | **69.40** | **74.89** | **75.14** | **70.85** | **79.43** | **76.37** | **69.77** | **82.97** |
| 10 | 73.10 | 70.12 | 76.09 | 76.48 | 71.71 | 81.24 | 76.74 | 71.55 | 81.93 |

### V3 Cascaded V2b

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 90.43 | 87.71 | 93.16 | 98.68 | 88.75 | 108.62 | 108.21 | 99.11 | 117.31 |
| 2 | 81.36 | 79.29 | 83.43 | 80.10 | 75.58 | 84.62 | 1491.28* | 1342.68 | 1639.87 |
| 3 | 75.24 | 73.61 | 76.86 | 76.98 | 72.48 | 81.49 | 97.91 | 92.52 | 103.31 |
| 4 | 76.19 | 73.89 | 78.50 | 77.35 | 72.64 | 82.07 | 105.15 | 102.30 | 108.01 |
| 5 | 74.87 | 74.11 | 75.62 | 75.25 | 72.23 | 78.27 | 103.47 | 84.08 | 122.86 |
| 6 | 72.73 | 71.01 | 74.45 | 75.37 | 70.20 | 80.55 | 83.54 | 78.40 | 88.68 |
| 7 | 74.91 | 70.75 | 79.07 | 74.77 | 70.04 | 79.51 | **79.59** | **73.20** | **85.99** |
| 8 | 73.47 | 70.74 | 76.19 | 74.64 | 69.31 | 79.96 | 80.65 | 75.10 | 86.21 |
| **9** | **72.15** | **68.93** | **75.36** | **74.19** | **69.03** | **79.35** | 80.66 | 74.21 | 87.12 |
| 10 | 73.38 | 69.89 | 76.87 | 74.87 | 69.49 | 80.24 | 80.47 | 73.24 | 87.70 |

*No HMD epoch 2: training instability spike (1491mm), recovered by epoch 3

### V3 Cross-Architecture Comparison

| Model | Single-Stage | Cascaded V2b | Delta |
|-------|-------------|-------------|-------|
| Ground Info | 72.14mm | 72.15mm | +0.01mm (same) |
| Baseline | 75.14mm | **74.19mm** | -0.95mm (-1.3%) |
| No-HMD | **76.37mm** | 79.59mm | +3.22mm (+4.2%) |

---

## V3 Blend Dataset Results

### V3 Blend Single-Stage

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 86.98 | 81.06 | 92.89 | 93.68 | 84.21 | 103.16 | 103.50 | 95.17 | 111.83 |
| 2 | 84.45 | 78.11 | 90.78 | 84.06 | 75.23 | 92.89 | 83.43 | 81.08 | 85.77 |
| 3 | 76.48 | 73.30 | 79.66 | 82.86 | 77.92 | 87.80 | 84.20 | 83.00 | 85.40 |
| 4 | 75.34 | 73.72 | 76.96 | 81.10 | 76.45 | 85.75 | 80.84 | 75.85 | 85.83 |
| 5 | 75.36 | 73.39 | 77.33 | 80.03 | 74.23 | 85.84 | 82.86 | 77.74 | 87.97 |
| 6 | 73.30 | 69.84 | 76.76 | 75.42 | 70.20 | 80.65 | 77.16 | 72.94 | 81.37 |
| 7 | 73.45 | 71.59 | 75.31 | 76.92 | 71.72 | 82.11 | 77.44 | 72.90 | 81.98 |
| 8 | 73.32 | 70.32 | 76.33 | 77.54 | 71.67 | 83.41 | 79.93 | 73.79 | 86.08 |
| **9** | **71.52** | **69.72** | **73.31** | **74.72** | **69.35** | **80.09** | **73.98** | **69.32** | **78.63** |
| 10 | 73.63 | 70.16 | 77.11 | 76.46 | 71.38 | 81.54 | 77.44 | 72.03 | 82.86 |

### V3 Blend Cascaded V2b

| Epoch | Ground Info (12-dim) ||| Baseline (9-dim) ||| No HMD |||
|-------|------|------|------|------|------|------|------|------|------|
| | Full | Upper | Lower | Full | Upper | Lower | Full | Upper | Lower |
| 1 | 93.62 | 88.06 | 99.19 | 92.01 | 81.79 | 102.22 | 94.00 | 94.98 | 93.01 |
| 2 | 77.90 | 76.17 | 79.63 | 81.03 | 75.75 | 86.31 | 93.76 | 88.45 | 99.07 |
| 3 | 79.90 | 78.49 | 81.30 | 78.55 | 73.64 | 83.45 | 87.42 | 85.30 | 89.54 |
| 4 | 75.54 | 72.31 | 78.77 | 79.85 | 75.28 | 84.42 | 100.27 | 93.52 | 107.01 |
| 5 | 74.37 | 71.63 | 77.11 | 79.37 | 75.18 | 83.55 | 89.83 | 80.10 | 99.56 |
| 6 | 72.00 | 70.48 | 73.52 | 75.33 | 70.26 | 80.40 | 80.56 | 75.74 | 85.38 |
| 7 | 72.47 | 68.63 | 76.31 | 75.84 | 70.89 | 80.80 | 80.63 | 72.54 | 88.73 |
| 8 | 71.89 | 68.60 | 75.19 | 75.62 | 70.79 | 80.46 | 80.63 | 73.90 | 87.35 |
| **9** | **70.30** | **67.34** | **73.26** | **73.71** | **68.33** | **79.08** | **76.49** | **70.39** | **82.60** |
| 10 | 71.40 | 67.86 | 74.93 | 74.62 | 69.36 | 79.88 | 77.74 | 70.70 | 84.78 |

### V3 Blend Cross-Architecture Comparison

| Model | Single-Stage | Cascaded V2b | Delta |
|-------|-------------|-------------|-------|
| Ground Info | 71.52mm | **70.30mm** | -1.22mm (-1.7%) |
| Baseline | 74.72mm | **73.71mm** | -1.01mm (-1.4%) |
| No-HMD | **73.98mm** | 76.49mm | +2.51mm (+3.4%) |

---

## V5 Flag Dataset Results (`egodataset_flag_fixed_ver5`)

Dataset: `egodataset_flag_fixed_ver5` with `use_2d_visible=True` — skips frames where all 2D joints are invisible (tracking error frames, ~2% of data). Same 8:2 batch split as V3/Blend.

### V5 Flag Single-Stage

#### Ground Info 12-dim (Best: ep10, 68.77mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 89.23 | 87.58 | 90.89 |
| 2 | 75.44 | 74.97 | 75.92 |
| 3 | 71.53 | 67.42 | 75.65 |
| 4 | 73.50 | 70.62 | 76.37 |
| 5 | 69.90 | 65.84 | 73.96 |
| 6 | 69.83 | 65.23 | 74.43 |
| 7 | 69.47 | 64.66 | 74.27 |
| 8 | 68.90 | 65.34 | 72.45 |
| 9 | 70.63 | 65.08 | 76.18 |
| **10** | **68.77** | **64.52** | **73.02** |

#### Baseline 9-dim (Best: ep8, 70.69mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 89.15 | 82.85 | 95.45 |
| 2 | 80.24 | 76.70 | 83.79 |
| 3 | 78.82 | 70.81 | 86.83 |
| 4 | 74.78 | 70.29 | 79.26 |
| 5 | 75.08 | 68.88 | 81.27 |
| 6 | 71.59 | 65.79 | 77.38 |
| 7 | 74.43 | 65.99 | 82.87 |
| **8** | **70.69** | **64.23** | **77.15** |
| 9 | 72.33 | 64.75 | 79.90 |
| 10 | 71.98 | 64.83 | 79.13 |

#### No-HMD (Best: ep9, 69.41mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 96.11 | 96.08 | 96.14 |
| 2 | 82.34 | 77.20 | 87.47 |
| 3 | 85.85 | 77.53 | 94.17 |
| 4 | 76.92 | 75.69 | 78.15 |
| 5 | 83.75 | 80.11 | 87.39 |
| 6 | 73.13 | 66.66 | 79.60 |
| 7 | 74.44 | 67.99 | 80.90 |
| 8 | 72.51 | 65.00 | 80.01 |
| **9** | **69.41** | **64.07** | **74.75** |
| 10 | 70.46 | 64.28 | 76.63 |

### V5 Flag Cascaded V2b

#### Ground Info 12-dim (Best: ep10, 64.36mm) — NEW BEST

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 82.91 | 85.24 | 80.59 |
| 2 | 72.82 | 75.26 | 70.38 |
| 3 | 66.82 | 66.08 | 67.56 |
| 4 | 67.07 | 66.68 | 67.45 |
| 5 | 68.05 | 64.62 | 71.49 |
| 6 | 66.65 | 62.27 | 71.02 |
| 7 | 66.41 | 63.04 | 69.79 |
| 8 | 65.53 | 61.80 | 69.26 |
| 9 | 65.23 | 60.61 | 69.86 |
| **10** | **64.36** | **60.91** | **67.81** |

#### Baseline 9-dim (Best: ep9, 70.97mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 86.46 | 85.48 | 87.43 |
| 2 | 84.92 | 82.18 | 87.67 |
| 3 | 76.43 | 68.68 | 84.17 |
| 4 | 74.43 | 70.49 | 78.36 |
| 5 | 77.76 | 70.11 | 85.42 |
| 6 | 73.78 | 65.73 | 81.83 |
| 7 | 72.92 | 65.21 | 80.62 |
| 8 | 71.44 | 65.14 | 77.74 |
| **9** | **70.97** | **63.86** | **78.08** |
| 10 | 71.21 | 64.41 | 78.01 |

#### No-HMD (Best: ep9, 71.79mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 108.42 | 106.06 | 110.78 |
| 2 | 110.93 | 94.24 | 127.62 |
| 3 | 110.73 | 95.80 | 125.66 |
| 4 | 114.12 | 104.74 | 123.51 |
| 5 | 99.68 | 100.10 | 99.27 |
| 6 | 76.26 | 71.85 | 80.68 |
| 7 | 76.75 | 75.19 | 78.31 |
| 8 | 75.21 | 67.54 | 82.88 |
| **9** | **71.79** | **65.40** | **78.19** |
| 10 | 71.91 | 65.28 | 78.54 |

Note: Cascaded No-HMD had severe convergence issues (epochs 1-5: 99-114mm), recovering from epoch 6. Still improving at epoch 10.

### V5 Flag Cross-Architecture Comparison

| Model | Single-Stage | Cascaded V2b | Delta |
|-------|-------------|-------------|-------|
| Ground Info | 68.77mm | **64.36mm** | **-4.41mm (-6.4%)** |
| Baseline | **70.69mm** | 70.97mm | +0.28mm (+0.4%) |
| No-HMD | **69.41mm** | 71.79mm | +2.38mm (+3.4%) |

Cascaded architecture strongly benefits Ground Info (12-dim), outperforming single-stage by 4.41mm. Without Ground Info, single-stage is competitive or better.

### V5 Flag Per-Action Breakdown (Best Epoch)

| Action | SS GI (ep10) | SS BL (ep8) | SS NoHMD (ep9) | Casc GI (ep10) | Casc BL (ep9) | Casc NoHMD (ep9) | Best |
|--------|-------------|------------|---------------|---------------|--------------|-----------------|------|
| Dancing1 | 73.25 | 74.27 | 72.31 | **68.38** | 77.44 | 77.70 | Casc GI |
| Dancing2 | 63.01 | 64.35 | 63.40 | **60.36** | 64.63 | 67.09 | Casc GI |
| Dancing3 | 64.43 | 64.88 | 65.36 | **61.00** | 67.61 | 68.65 | Casc GI |
| Gaming-Archery | 80.10 | 81.86 | 82.69 | **78.82** | 80.59 | 87.45 | Casc GI |
| Gaming-Baseball | 114.48 | 118.00 | **111.25** | 109.02 | 112.27 | 112.63 | **Casc GI** |
| Gaming-Boxing | 76.34 | 81.51 | 76.24 | **72.75** | 79.85 | 77.43 | Casc GI |
| Gaming-Golf | 85.07 | 87.05 | 84.18 | **82.99** | 86.83 | 87.18 | Casc GI |
| Gaming-Shooting | 73.80 | 74.72 | 73.35 | **68.31** | 73.59 | 74.67 | Casc GI |
| Greeting-ShakingHand | 56.44 | 60.05 | 59.93 | **54.58** | 62.08 | 62.48 | Casc GI |
| Greeting-WavingHand | 56.73 | 55.18 | 55.35 | **48.57** | 57.43 | 59.29 | Casc GI |
| Patting | 61.63 | 60.02 | 62.20 | **54.34** | 65.48 | 64.02 | Casc GI |
| Reacting-Cheering | 64.52 | 63.97 | 66.06 | **57.83** | 69.36 | 68.72 | Casc GI |
| Reacting-Clapping | 93.46 | 97.23 | 97.29 | **90.67** | 96.62 | 96.24 | Casc GI |
| Reacting-Yelling | 50.96 | 51.99 | 52.32 | **47.00** | 53.66 | 55.57 | Casc GI |
| Talking | 60.49 | 60.20 | **56.75** | 54.44 | 61.29 | 62.27 | **Casc GI** |
| UpperStreching | 58.05 | 61.67 | 59.54 | **51.93** | 60.51 | 60.43 | Casc GI |
| Walking | 61.02 | 62.27 | 62.33 | **58.71** | 63.93 | 65.28 | Casc GI |
| Workout-BicelCurl | 52.61 | 57.12 | 53.41 | **46.79** | 56.42 | 55.54 | Casc GI |
| Workout-FrontRaise | 57.36 | 60.44 | 60.16 | **52.75** | 61.32 | 60.71 | Casc GI |
| Workout-KettleBell | 70.70 | 75.36 | 72.77 | **66.28** | 68.05 | 71.56 | Casc GI |

**Cascaded Ground Info wins ALL 20 actions.** Largest improvements vs next best:
- Greeting-WavingHand: 48.57mm (-6.61mm vs SS Baseline)
- Patting: 54.34mm (-5.68mm vs SS Baseline)
- Reacting-Cheering: 57.83mm (-6.14mm vs SS Baseline)
- UpperStreching: 51.93mm (-6.12mm vs SS GI)

### V5 Flag vs V3 Blend Comparison

| Model | V3 Blend | V5 Flag | Improvement |
|-------|----------|---------|-------------|
| SS Ground Info | 71.52mm | **68.77mm** | -2.75mm (-3.8%) |
| SS Baseline | 74.72mm | **70.69mm** | -4.03mm (-5.4%) |
| SS No-HMD | 73.98mm | **69.41mm** | -4.57mm (-6.2%) |
| Casc Ground Info | 70.30mm | **64.36mm** | **-5.94mm (-8.5%)** |
| Casc Baseline | 73.71mm | **70.97mm** | -2.74mm (-3.7%) |
| Casc No-HMD | 76.49mm | **71.79mm** | -4.70mm (-6.1%) |

Removing ~2% error frames from the V5 Flag dataset dramatically improves all configurations, with the biggest gain in Cascaded Ground Info (-5.94mm).

---

## V3 Blend Cascaded 20-Epoch Results

Settings: milestones=[10, 16], gamma=0.5 (doubled from 10ep schedule)

### Per-Epoch Results

#### Cascaded Ground Info 12-dim (20ep)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 86.80 | 85.51 | 88.10 |
| 2 | 81.56 | 75.63 | 87.50 |
| 3 | 77.64 | 74.27 | 81.00 |
| 4 | 77.94 | 73.21 | 82.67 |
| 5 | 74.30 | 71.98 | 76.62 |
| 6 | 74.76 | 73.18 | 76.35 |
| 7 | 80.91 | 79.03 | 82.80 |
| 8 | 74.07 | 70.40 | 77.74 |
| 9 | 74.63 | 70.09 | 79.17 |
| 10 | 74.41 | 72.23 | 76.59 |
| **11** | **72.40** | **67.58** | **77.21** |
| 12 | 73.34 | 68.54 | 78.14 |
| 13 | 72.53 | 68.08 | 76.98 |
| 14 | 75.55 | 70.55 | 80.55 |
| 15 | 72.45 | 68.07 | 76.84 |
| 16 | 72.73 | 68.43 | 77.04 |
| 17 | 72.91 | 68.23 | 77.60 |
| 18 | 73.34 | 68.12 | 78.56 |
| 19 | 73.22 | 68.35 | 78.09 |
| 20 | 72.79 | 67.78 | 77.80 |

#### Cascaded Baseline 9-dim (20ep)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 104.69 | 92.19 | 117.19 |
| 2 | 83.08 | 77.06 | 89.11 |
| 3 | 79.06 | 74.02 | 84.10 |
| 4 | 79.78 | 73.99 | 85.56 |
| 5 | 78.44 | 74.40 | 82.49 |
| 6 | 77.03 | 72.06 | 82.00 |
| 7 | 76.97 | 72.60 | 81.35 |
| 8 | 74.33 | 69.45 | 79.21 |
| 9 | 75.17 | 69.84 | 80.51 |
| 10 | 76.67 | 70.58 | 82.76 |
| 11 | 73.56 | 67.54 | 79.59 |
| 12 | 74.67 | 67.97 | 81.38 |
| 13 | 74.05 | 68.81 | 79.30 |
| 14 | 73.42 | 68.20 | 78.65 |
| **15** | **73.34** | **68.31** | **78.36** |
| 16 | 73.63 | 68.17 | 79.10 |
| 17 | 73.86 | 67.72 | 80.01 |
| 18 | 73.44 | 67.76 | 79.12 |
| 19 | 73.81 | 68.11 | 79.51 |
| 20 | 73.89 | 67.83 | 79.95 |

#### Cascaded No-HMD (20ep)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 123.80 | 108.04 | 139.56 |
| 2 | 128.86 | 113.55 | 144.17 |
| 3 | 96.82 | 95.52 | 98.12 |
| 4 | 113.02 | 109.35 | 116.70 |
| 5 | 161.17 | 179.91 | 142.43 |
| 6 | 104.79 | 119.54 | 90.05 |
| 7 | 92.50 | 83.23 | 101.77 |
| 8 | 87.34 | 83.36 | 91.32 |
| 9 | 86.59 | 76.88 | 96.29 |
| 10 | 82.26 | 78.90 | 85.62 |
| 11 | 77.48 | 70.41 | 84.56 |
| 12 | 77.52 | 72.13 | 82.90 |
| 13 | 78.14 | 71.95 | 84.32 |
| 14 | 79.46 | 73.33 | 85.60 |
| 15 | 77.94 | 72.16 | 83.73 |
| 16 | 79.02 | 73.33 | 84.71 |
| 17 | 76.97 | 71.00 | 82.94 |
| 18 | 76.27 | 71.02 | 81.52 |
| 19 | 76.94 | 71.02 | 82.86 |
| **20** | **76.14** | **70.85** | **81.43** |

Note: No-HMD had severe instability in early epochs (ep2: 128.86, ep5: 161.17) but recovered and was still improving at epoch 20.

### Per-Action Breakdown (Best Epoch)

| Action | Ground Info (ep11) | Baseline (ep15) | No HMD (ep20) | Best Model |
|--------|-------------------|-----------------|---------------|------------|
| Dancing1 | 70.60 | **69.83** | 73.49 | Baseline |
| Dancing2 | **64.47** | 64.68 | 69.45 | Ground Info |
| Dancing3 | 62.92 | **62.37** | 64.73 | Baseline |
| Gaming-Archery | **77.27** | 79.88 | 87.76 | Ground Info |
| Gaming-Baseball | **112.00** | 112.03 | 115.84 | Ground Info |
| Gaming-Boxing | **80.42** | 80.51 | 84.77 | Ground Info |
| Gaming-Golf | **85.90** | 86.79 | 92.92 | Ground Info |
| Gaming-Shooting | **78.48** | 78.67 | 83.95 | Ground Info |
| Greeting-ShakingHand | **59.45** | 60.69 | 60.98 | Ground Info |
| Greeting-WavingHand | **64.01** | 64.04 | 65.61 | Ground Info |
| Patting | **78.04** | 78.08 | 80.69 | Ground Info |
| Reacting-Cheering | **68.85** | 71.25 | 74.02 | Ground Info |
| Reacting-Clapping | **94.69** | 98.04 | 101.04 | Ground Info |
| Reacting-Yelling | 54.21 | **53.70** | 56.29 | Baseline |
| Talking | **68.28** | 68.46 | 70.02 | Ground Info |
| UpperStreching | **57.24** | 60.27 | 62.96 | Ground Info |
| Walking | **61.44** | 62.87 | 66.99 | Ground Info |
| Workout-BicelCurl | 53.75 | 54.55 | **51.85** | No HMD |
| Workout-FrontRaise | **69.32** | 69.93 | 70.06 | Ground Info |
| Workout-KettleBell | **86.96** | 90.00 | 89.61 | Ground Info |

**Ground Info wins 16/20 actions**, Baseline wins 3 (Dancing1, Dancing3, Reacting-Yelling), No-HMD wins 1 (Workout-BicelCurl).

### Per-Action Analysis

**Hardest actions** (highest MPJPE):
- Gaming-Baseball: 112.00mm — wide arm swings, high spatial variance
- Reacting-Clapping: 94.69mm — rapid repetitive motions
- Workout-KettleBell: 86.96mm — large lower body range of motion

**Easiest actions** (lowest MPJPE):
- Workout-BicelCurl: 53.75mm — stationary, limited range
- Reacting-Yelling: 54.21mm — mostly upper body, minimal movement
- UpperStreching: 57.24mm — slow, predictable motion

**Ground Info biggest advantage** (vs Baseline):
- Reacting-Clapping: -3.35mm (94.69 vs 98.04)
- UpperStreching: -3.03mm (57.24 vs 60.27)
- Gaming-Archery: -2.61mm (77.27 vs 79.88)
- Reacting-Cheering: -2.40mm (68.85 vs 71.25)

### 10ep vs 20ep Comparison

| Model | 10ep Best | 20ep Best | Delta | Note |
|-------|-----------|-----------|-------|------|
| Ground Info | **70.30** (ep9) | 72.40 (ep11) | +2.10mm | 10ep better (different random seed/run) |
| Baseline | **73.71** (ep9) | 73.34 (ep15) | -0.37mm | Slight improvement |
| No-HMD | **76.49** (ep9) | 76.14 (ep20) | -0.35mm | Still improving, may benefit from 30ep |

The 20ep run did not improve Ground Info over the 10ep run's best (70.30mm at ep9). This suggests the 10ep result may have benefited from favorable training dynamics in that particular run. Baseline and No-HMD showed marginal improvements with extended training.

---

## V5 Flag Architecture Ablation Study

Ablation of the Cascaded architecture on V5 Flag dataset. All use `CustomEgoposeCascadedRefinementHead_enhanced` with different flags.

### Paper TABLE V: Single-stage vs Full Model × LHF vs GBH

| Architecture | LHF (9-dim) | GBH (12-dim) | GBH Gain |
|-------------|-------------|--------------|----------|
| Single-stage | 70.24mm (ep8) | 69.33mm (ep6) | -0.91mm (-1.3%) |
| Full model (+ Stage 2) | 70.97mm (ep9) | **64.36mm** (ep10) | **-6.61mm (-9.3%)** |
| Stage 2 Gain | -0.73mm (-1.0%) | **-4.97mm (-7.2%)** | |

**Key insight:** Stage 2 refinement and GBH are synergistic — GBH alone gives -0.91mm, Stage 2 alone gives -0.73mm, but combined they give -6.61mm. The refinement MLP can exploit ground-based height cues that the Stage 1 linear decoder cannot.

### Full Ablation Summary

| Variant | Description | Best Epoch | Full Body | Upper Body | Lower Body | vs Full Model |
|---------|-------------|-----------|-----------|------------|------------|---------------|
| **Full Model (GBH)** | Stage 1 + Stage 2 + Aux Decoders, 12-dim | 10 | **64.36mm** | **60.91mm** | **67.81mm** | — (baseline) |
| **No Aux Decoder** | Stage 1 + Stage 2, no heatmap/HMD recon | 9 | 65.33mm | 62.08mm | 68.58mm | +0.97mm (+1.5%) |
| **Stage 1 Only (GBH)** | Stage 1 + Aux Decoders, 12-dim, no refinement | 6 | 69.33mm | 65.33mm | 73.34mm | +4.97mm (+7.7%) |
| **Stage 1 Only (LHF)** | Stage 1 + Aux Decoders, 9-dim, no refinement | 8 | 70.24mm | 64.57mm | 75.91mm | +5.88mm (+9.1%) |

### Ablation: Stage 1 Only (no refinement)

Uses `use_refinement=False` — disables Stage 2 (per-joint grid sampling + kinematic features + HMD context refinement MLP).

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 85.85 | 82.32 | 89.38 |
| 2 | 75.68 | 74.98 | 76.37 |
| 3 | 75.44 | 69.10 | 81.78 |
| 4 | 72.78 | 69.57 | 76.00 |
| 5 | 71.99 | 68.27 | 75.71 |
| **6** | **69.33** | **65.33** | **73.34** |
| 7 | 70.20 | 65.16 | 75.24 |
| 8 | 69.89 | 65.42 | 74.36 |
| 9 | 70.78 | 64.12 | 77.44 |
| 10 | 70.11 | 64.93 | 75.28 |

Per-action breakdown (Epoch 6):

| Action | Full Body | Upper Body | Lower Body |
|--------|-----------|------------|------------|
| Dancing | 66.37 | 62.23 | 70.52 |
| Gaming | 84.42 | 79.75 | 89.08 |
| Greeting | 59.17 | 53.08 | 65.26 |
| Patting | 61.30 | 57.91 | 64.69 |
| Reacting | 72.35 | 71.64 | 73.05 |
| Talking | 59.40 | 56.76 | 62.05 |
| UpperStreching | 58.84 | 58.62 | 59.06 |
| Walking | 64.58 | 55.10 | 74.06 |
| Workout | 62.16 | 57.53 | 66.78 |

### Ablation: Stage 1 Only + LHF 9-dim (no refinement, no ground heights)

Uses `use_refinement=False` + `hmd_info_size=9` + `ground_info_mode=None`.

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 101.79 | 95.98 | 107.59 |
| 2 | 80.94 | 77.61 | 84.26 |
| 3 | 77.92 | 70.72 | 85.11 |
| 4 | 76.25 | 69.20 | 83.30 |
| 5 | 73.37 | 67.27 | 79.47 |
| 6 | 71.75 | 65.33 | 78.17 |
| 7 | 73.67 | 65.69 | 81.65 |
| **8** | **70.24** | **64.57** | **75.91** |
| 9 | 71.61 | 64.89 | 78.32 |
| 10 | 70.23 | 64.78 | 75.67 |

Per-action breakdown (Epoch 8):

| Action | Full Body | Upper Body | Lower Body |
|--------|-----------|------------|------------|
| Dancing | 69.37 | 62.57 | 76.16 |
| Gaming | 86.76 | 80.73 | 92.80 |
| Greeting | 57.43 | 50.14 | 64.71 |
| Patting | 60.08 | 58.21 | 61.96 |
| Reacting | 71.60 | 70.25 | 72.95 |
| Talking | 60.22 | 53.62 | 66.81 |
| UpperStreching | 59.34 | 55.37 | 63.31 |
| Walking | 63.46 | 52.45 | 74.47 |
| Workout | 63.36 | 56.76 | 69.96 |

### Ablation: No Auxiliary Decoders

Uses `use_auxiliary_decoders=False` — disables heatmap reconstruction decoder (`loss_heatmap_recon`, weight=500) and HMD reconstruction loss (`loss_hmd`). Stage 2 refinement is still active.

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 96.97 | 96.07 | 97.88 |
| 2 | 75.21 | 77.88 | 72.54 |
| 3 | 78.51 | 77.57 | 79.46 |
| 4 | 70.96 | 69.42 | 72.50 |
| 5 | 67.63 | 65.59 | 69.67 |
| 6 | 67.09 | 65.94 | 68.25 |
| 7 | 66.43 | 63.46 | 69.39 |
| 8 | 66.13 | 62.85 | 69.42 |
| **9** | **65.33** | **62.08** | **68.58** |
| 10 | 65.87 | 63.31 | 68.43 |

Per-action breakdown (Epoch 9):

| Action | Full Body | Upper Body | Lower Body |
|--------|-----------|------------|------------|
| Dancing | 64.04 | 60.46 | 67.61 |
| Gaming | 82.50 | 78.35 | 86.64 |
| Greeting | 53.09 | 47.99 | 58.20 |
| Patting | 57.97 | 58.49 | 57.45 |
| Reacting | 66.42 | 66.94 | 65.89 |
| Talking | 55.32 | 52.03 | 58.62 |
| UpperStreching | 52.45 | 52.34 | 52.56 |
| Walking | 58.46 | 51.57 | 65.35 |
| Workout | 57.57 | 52.82 | 62.32 |

### Ablation Analysis

**Stage 2 Refinement contributes +4.97mm (7.7%) improvement:**
- Full Body: 69.33 → 64.36mm
- Upper Body: 65.33 → 60.91mm (-4.42mm)
- Lower Body: 73.34 → 67.81mm (-5.53mm)
- Lower body benefits more from refinement (+5.53mm vs +4.42mm), consistent with ground info helping depth estimation

**Auxiliary Decoders contribute +0.97mm (1.5%) improvement:**
- Full Body: 65.33 → 64.36mm
- Upper Body: 62.08 → 60.91mm (-1.17mm)
- Lower Body: 68.58 → 67.81mm (-0.77mm)
- Heatmap reconstruction + HMD reconstruction provide modest but consistent regularization

**Convergence behavior:**
- Stage 1 Only: peaks early (epoch 6), slight degradation afterward
- No Aux Decoder: slower start (ep1: 96.97 vs full model 82.91), converges later (ep9)
- Full Model: steady improvement through epoch 10, still not plateaued

**Per-action comparison (Full Body, best epoch):**

| Action | Full Model GBH (ep10) | No Aux (ep9) | Stage1 GBH (ep6) | Stage1 LHF (ep8) | Refine Δ (GBH) | GBH Δ (Stage1) |
|--------|----------------------|-------------|-------------------|-------------------|----------------|----------------|
| Dancing | 63.25* | 64.04 | 66.37 | 69.37 | -3.12 | -3.00 |
| Gaming | 82.38* | 82.50 | 84.42 | 86.76 | -2.04 | -2.34 |
| Greeting | 51.57* | 53.09 | 59.17 | 57.43 | -7.60 | +1.74 |
| Patting | 54.34 | 57.97 | 61.30 | 60.08 | -6.96 | +1.22 |
| Reacting | 65.17* | 66.42 | 72.35 | 71.60 | -7.18 | +0.75 |
| Talking | 54.44 | 55.32 | 59.40 | 60.22 | -4.96 | -0.82 |
| UpperStreching | 51.93 | 52.45 | 58.84 | 59.34 | -6.91 | -0.50 |
| Walking | 58.71 | 58.46 | 64.58 | 63.46 | -5.87 | +1.12 |
| Workout | 55.28* | 57.57 | 62.16 | 63.36 | -6.88 | -1.20 |

*Full Model uses individual action values averaged across sub-actions from detailed per-action table.

**Key insights:**
- **Refinement** provides large, consistent gains across all actions (2-8mm), with the largest on Greeting (-7.60mm) and Reacting (-7.18mm)
- **Auxiliary decoders** provide smaller gains (0-4mm), largest on Patting (-3.63mm) and Workout (-2.29mm)
- **GBH at Stage 1 level** shows mixed results per-action: helps Dancing (-3.00mm), Gaming (-2.34mm), Workout (-1.20mm) but slightly hurts Greeting (+1.74mm), Patting (+1.22mm) — overall -0.91mm
- **GBH + Refinement synergy**: the per-joint refinement MLP can leverage height cues far better than the Stage 1 linear decoder, explaining the dramatic combined gain (-6.61mm)

---

## Key Findings

### 1. V3 Blend is the best dataset

| Config | V4 | V3 Original | V3 Blend | V3→Blend Improvement |
|--------|-----|-------------|----------|---------------------|
| Single GI | 97.16 | 72.14 | **71.52** | -0.62mm (-0.9%) |
| Single BL | 102.50 | 75.14 | **74.72** | -0.42mm (-0.6%) |
| Single NoHMD | 107.83 | 76.37 | **73.98** | -2.39mm (-3.1%) |
| Cascaded GI | 98.57 | 72.15 | **70.30** | -1.85mm (-2.6%) |
| Cascaded BL | 106.03 | 74.19 | **73.71** | -0.48mm (-0.6%) |
| Cascaded NoHMD | 107.79 | 79.59 | **76.49** | -3.10mm (-3.9%) |

The blend dataset consistently improves all configurations. Largest gains in No-HMD models, suggesting the blend preprocessing provides better visual features.

### 2. Ground Info consistently provides the best results

Across all datasets and architectures, 12-dim Ground Info HMD mode achieves the lowest MPJPE:

| Dataset | Best Ground Info | Best Baseline | GI Advantage |
|---------|-----------------|---------------|--------------|
| V4 | 97.16mm | 102.50mm | -5.34mm (-5.2%) |
| V3 | 72.14mm | 74.19mm | -2.05mm (-2.8%) |
| V3 Blend | **70.30mm** | 73.71mm | -3.41mm (-4.6%) |

### 3. Architecture preference depends on dataset size/quality

| Dataset | Single better? | Cascaded better? | Notes |
|---------|---------------|-----------------|-------|
| **V4** (28k samples) | **Yes** (all 3 configs) | - | Small dataset favors simpler model |
| **V3** (larger) | Tie (GI/NoHMD) | Tie (BL) | Similar performance |
| **V3 Blend** (larger) | - (NoHMD only) | **Yes** (GI/BL) | Blend + cascaded = best combo |

On V4 (smallest dataset), single-stage outperforms cascaded. On V3 Blend (best dataset), cascaded edges ahead for HMD-equipped models.

### 4. Cascaded No-HMD is unstable

| Dataset | Cascaded No-HMD Issues |
|---------|----------------------|
| V4 | Epoch 1 spike (297mm) |
| V3 | Epoch 2 spike (1491mm) |
| V3 Blend | Stable (best: 76.49mm) |
| V5 Flag | Epochs 1-5 unstable (99-114mm), recovered ep6 |

The cascaded architecture without HMD guidance struggles to converge, especially on smaller/noisier datasets.

### 5. Upper/Lower Body balance with Ground Info

| Dataset | Model | Upper | Lower | Gap |
|---------|-------|-------|-------|-----|
| V4 Single GI | 96.73 | 97.60 | 0.87mm |
| V3 Blend Cascaded GI | 67.34 | 73.26 | 5.92mm |
| **V5 Flag Cascaded GI** | **60.91** | **67.81** | **6.90mm** |
| V5 Flag Cascaded BL | 63.86 | 78.08 | 14.22mm |
| V5 Flag Cascaded NoHMD | 65.40 | 78.19 | 12.79mm |

Ground Info reduces the Upper-Lower gap from 12.79mm (NoHMD) to 6.90mm. The V5 Flag Cascaded GI achieves the best absolute upper body (60.91mm) and lower body (67.81mm) of any Kinect experiment.

### 6. Error frame filtering (V5 Flag) has dramatic impact

Removing ~2% tracking error frames improved all configurations by 2.7-5.9mm:

| Model | V3 Blend | V5 Flag | Delta |
|-------|----------|---------|-------|
| Casc GI | 70.30mm | **64.36mm** | **-5.94mm (-8.5%)** |
| SS No-HMD | 73.98mm | **69.41mm** | -4.57mm (-6.2%) |
| SS BL | 74.72mm | **70.69mm** | -4.03mm (-5.4%) |

This shows that even a small proportion of bad GT data significantly degrades model performance. Data quality > data quantity.

### 7. Cascaded Ground Info dominates all actions on V5 Flag

On V5 Flag, Cascaded Ground Info wins **ALL 20 actions** (vs V3 Blend where it won 16/20). The largest per-action gains vs next-best model:
- Greeting-WavingHand: 48.57mm (-6.61mm)
- Reacting-Cheering: 57.83mm (-6.14mm)
- UpperStreching: 51.93mm (-6.12mm)
- Patting: 54.34mm (-5.68mm)

### 8. Single-stage No-HMD anomaly persists

On both V3 Blend and V5 Flag, single-stage No-HMD outperforms single-stage Baseline:
- V3 Blend: 73.98 vs 74.72mm
- V5 Flag: 69.41 vs 70.69mm

This suggests the 9-dim HMD info may add noise in the single-stage architecture, while the cascaded architecture can better leverage it.

### 9. Extended training (20ep) shows diminishing returns

The 20ep cascaded runs did not substantially improve over 10ep. Ground Info actually performed worse (72.40 vs 70.30mm), likely due to run-to-run variance. Baseline improved marginally (-0.37mm). No-HMD was still improving at epoch 20, suggesting vision-only models need longer training to converge.

---

## Work Dirs

### V4

| Experiment | Path |
|-----------|------|
| Single Ground Info | `work_dirs/HMD_kinect_v4_ground_info_10ep/` |
| Single Baseline | `work_dirs/HMD_kinect_v4_baseline_10ep/` |
| Single No-HMD | `work_dirs/HMD_kinect_v4_no_hmd_10ep/` |
| Cascaded Ground Info | `work_dirs/HMD_kinect_v4_cascaded_ground_info_10ep/` |
| Cascaded Baseline | `work_dirs/HMD_kinect_v4_cascaded_baseline_10ep/` |
| Cascaded No-HMD | `work_dirs/HMD_kinect_v4_cascaded_no_hmd_10ep/` |

### V3 / V3 Blend (shared work_dirs, multiple runs)

| Experiment | Path | Runs |
|-----------|------|------|
| Single Ground Info | `work_dirs/HMD_kinect_v3_ground_info_10ep/` | V3 (20260312), Blend (20260313) |
| Single Baseline | `work_dirs/HMD_kinect_v3_baseline_10ep/` | V3 (20260312), Blend (20260313) |
| Single No-HMD | `work_dirs/HMD_kinect_v3_no_hmd_10ep/` | V3 (20260312), Blend (20260313) |
| Cascaded Ground Info | `work_dirs/HMD_kinect_v3_cascaded_ground_info_10ep/` | V3 (20260312), Blend (20260313, 20260314) |
| Cascaded Baseline | `work_dirs/HMD_kinect_v3_cascaded_baseline_10ep/` | V3 (20260312), Blend (20260313, 20260314) |
| Cascaded No-HMD | `work_dirs/HMD_kinect_v3_cascaded_no_hmd_10ep/` | V3 (20260312), Blend (20260313, 20260315) |

### V3 Blend Cascaded 20ep

| Experiment | Path |
|-----------|------|
| Ground Info | `work_dirs/HMD_kinect_v3_cascaded_ground_info_20ep/` |
| Baseline | `work_dirs/HMD_kinect_v3_cascaded_baseline_20ep/` |
| No-HMD | `work_dirs/HMD_kinect_v3_cascaded_no_hmd_20ep/` |

### V5 Flag

| Experiment | Path |
|-----------|------|
| Single Ground Info | `work_dirs/HMD_kinect_v5_flag_ground_info_10ep/` |
| Single Baseline | `work_dirs/HMD_kinect_v5_flag_baseline_10ep/` |
| Single No-HMD | `work_dirs/HMD_kinect_v5_flag_no_hmd_10ep/` |
| Cascaded Ground Info | `work_dirs/HMD_kinect_v5_flag_cascaded_ground_info_10ep/` |
| Cascaded Baseline | `work_dirs/HMD_kinect_v5_flag_cascaded_baseline_10ep/` |
| Cascaded No-HMD | `work_dirs/HMD_kinect_v5_flag_cascaded_no_hmd_10ep/` |

### V5 Flag Ablation Study

| Experiment | Path |
|-----------|------|
| Stage 1 Only + GBH (no refinement) | `work_dirs/HMD_kinect_v5_flag_cascaded_stage1_only_ground_info_10ep/` |
| Stage 1 Only + LHF (no refinement, 9-dim) | `work_dirs/HMD_kinect_v5_flag_cascaded_stage1_only_baseline_10ep/` |
| Cascaded No Aux Decoder | `work_dirs/HMD_kinect_v5_flag_cascaded_no_aux_decoder_ground_info_10ep/` |

### WandB Projects

| Dataset | Projects |
|---------|----------|
| V3 | `kinect_v3-ground-info-12dim`, `kinect_v3-baseline-9dim`, `kinect_v3-no-hmd-vision-only`, `kinect_v3-cascaded-ground-info-12dim`, `kinect_v3-cascaded-baseline-9dim`, `kinect_v3-cascaded-no-hmd` |
| V3 Blend | `kinect_v3-blend-ground-info-12dim`, `kinect_v3-blend-baseline-9dim`, `kinect_v3-blend-no-hmd-vision-only`, `kinect_v3-blend-cascaded-ground-info-12dim`, `kinect_v3-blend-cascaded-baseline-9dim`, `kinect_v3-blend-cascaded-no-hmd` |
| V5 Flag | `egodataset_flag_fixed_ver5-ground-info-12dim`, `egodataset_flag_fixed_ver5-baseline-9dim`, `egodataset_flag_fixed_ver5-no-hmd-vision-only`, `egodataset_flag_fixed_ver5-cascaded-ground-info-12dim`, `egodataset_flag_fixed_ver5-cascaded-baseline-9dim`, `egodataset_flag_fixed_ver5-cascaded-no-hmd` |

---

## Next Steps

- [x] ~~Train V3 Blend for 20 epochs~~ → Done. 20ep did not improve over 10ep best
- [x] ~~Add per-action breakdown metrics~~ → Done. Cascaded GI wins all 20 actions on V5 Flag
- [x] ~~Train V5 Flag (error frame filtering)~~ → Done. **New best: 64.36mm** (-5.94mm vs V3 Blend)
- [x] ~~Architecture ablation (Stage 1 Only, No Aux Decoder)~~ → Done. Refinement = +4.97mm (7.7%), Aux Decoders = +0.97mm (1.5%)
- [ ] Train V5 Flag Cascaded Ground Info for 20ep (still improving at ep10)
- [ ] Investigate single-stage No-HMD beating Baseline anomaly (persists across V3 Blend and V5 Flag)
- [ ] Try V3 architecture (best on xRegopose: 34.06mm) on Kinect data
- [ ] Combine V5 Flag filtering with blend preprocessing
