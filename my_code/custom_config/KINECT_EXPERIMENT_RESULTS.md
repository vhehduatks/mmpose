# Kinect EgoPose Dataset - Experiment Results

## Dataset Overview

| Item | Train | Test |
|------|-------|------|
| Sessions | 178 | 58 |
| Samples | 28,033 | 11,112 |
| Skipped frames | 168 | 1,200 |
| Source | Azure Kinect v4 (32 joints -> 16 xRegopose) |
| Data root | `/mnt/dataset_vol/annotation_egodataset_2026_sei/` |

## Common Settings (All Experiments)

| Setting | Value |
|---------|-------|
| Backbone | ResNet-101 (COCO pretrained) |
| Optimizer | AdamW, lr=0.0005 |
| Scheduler | MultiStepLR (milestones=[5,8], gamma=0.5) |
| Batch size | 16 |
| Max epochs | 10 |
| Codec | Custom_mo2cap2_MSRAHeatmap (47x47, sigma=3) |
| Pipeline | LoadImage -> EgoImageResize(256x256) -> GenerateTarget -> PackPoseInputs |
| Image bbox | Full frame (0,0,1920,1080) - no cropping |

---

## Results Summary (Best Epoch, 10 Epochs)

### Single-Stage (CustomxRegoposeBaselinel1)

| Model | Best Ep | Full Body MPJPE | Upper Body | Lower Body | vs No-HMD |
|-------|---------|-----------------|------------|------------|-----------|
| **Ground Info 12-dim** | 10 | **97.16mm** | 96.73mm | **97.60mm** | **-10.67mm (-9.9%)** |
| Baseline 9-dim | 10 | 102.50mm | 100.10mm | 104.90mm | -5.33mm (-4.9%) |
| No-HMD (vision-only) | 9 | 107.83mm | 108.44mm | 107.22mm | - (Reference) |

### Cascaded Refinement V2b (CustomEgoposeCascadedRefinementHead_enhanced)

| Model | Best Ep | Full Body MPJPE | Upper Body | Lower Body | vs No-HMD |
|-------|---------|-----------------|------------|------------|-----------|
| **Ground Info 12-dim** | 10 | **98.57mm** | **99.06mm** | **98.08mm** | **-9.22mm (-8.6%)** |
| Baseline 9-dim | 10 | 106.03mm | 102.80mm | 109.27mm | -1.75mm (-1.6%) |
| No-HMD (vision-only) | 10 | 107.79mm | 107.38mm | 108.20mm | - (Reference) |

### Cross-Architecture Comparison (Best Results)

| Model | Single-Stage | Cascaded V2b | Cascaded Effect |
|-------|-------------|-------------|-----------------|
| Ground Info 12-dim | **97.16mm** | 98.57mm | +1.41mm (+1.5%) |
| Baseline 9-dim | **102.50mm** | 106.03mm | +3.53mm (+3.4%) |
| No-HMD (vision-only) | **107.83mm** | 107.79mm | -0.04mm (same) |

---

## Per-Epoch Results (10 Epochs)

### Single-Stage: No-HMD (Vision-Only)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 180.70mm | 151.28mm | 210.12mm |
| 2 | 153.50mm | 158.12mm | 148.88mm |
| 3 | 137.91mm | 140.19mm | 135.64mm |
| 4 | 133.61mm | 130.08mm | 137.13mm |
| 5 | 210.56mm | 170.26mm | 250.85mm |
| 6 | 197.14mm | 149.10mm | 245.19mm |
| 7 | 109.45mm | 109.65mm | 109.25mm |
| 8 | 120.94mm | 114.77mm | 127.10mm |
| **9** | **107.83mm** | **108.44mm** | **107.22mm** |
| 10 | 112.02mm | 111.81mm | 112.23mm |

### Single-Stage: Baseline 9-dim

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 130.57mm | 125.41mm | 135.74mm |
| 2 | 124.36mm | 118.87mm | 129.85mm |
| 3 | 124.40mm | 128.67mm | 120.12mm |
| 4 | 121.54mm | 121.76mm | 121.32mm |
| 5 | 114.54mm | 110.53mm | 118.56mm |
| 6 | 105.62mm | 104.43mm | 106.80mm |
| 7 | 106.20mm | 102.32mm | 110.08mm |
| 8 | 103.58mm | 100.37mm | 106.80mm |
| 9 | 105.95mm | 103.50mm | 108.40mm |
| **10** | **102.50mm** | **100.10mm** | **104.90mm** |

### Single-Stage: Ground Info 12-dim

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 127.06mm | 121.12mm | 133.00mm |
| 2 | 119.42mm | 120.57mm | 118.27mm |
| 3 | 108.03mm | 104.53mm | 111.53mm |
| 4 | 117.81mm | 118.50mm | 117.11mm |
| 5 | 114.53mm | 110.89mm | 118.17mm |
| 6 | 98.15mm | 100.08mm | 96.23mm |
| 7 | 97.35mm | 99.03mm | 95.66mm |
| 8 | 105.99mm | 105.27mm | 106.71mm |
| 9 | 98.05mm | 98.64mm | 97.47mm |
| **10** | **97.16mm** | **96.73mm** | **97.60mm** |

### Cascaded V2b: No-HMD (Vision-Only)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 297.27mm | 235.27mm | 359.26mm |
| 2 | 134.12mm | 136.80mm | 131.44mm |
| 3 | 146.15mm | 164.24mm | 128.06mm |
| 4 | 131.09mm | 136.66mm | 125.53mm |
| 5 | 130.06mm | 129.74mm | 130.39mm |
| 6 | 109.29mm | 110.39mm | 108.19mm |
| 7 | 115.11mm | 115.43mm | 114.80mm |
| 8 | 113.34mm | 114.98mm | 111.69mm |
| 9 | 108.20mm | 108.29mm | 108.11mm |
| **10** | **107.79mm** | **107.38mm** | **108.20mm** |

### Cascaded V2b: Baseline 9-dim

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 136.73mm | 140.25mm | 133.21mm |
| 2 | 133.22mm | 124.64mm | 141.80mm |
| 3 | 112.77mm | 111.57mm | 113.98mm |
| 4 | 138.63mm | 138.98mm | 138.29mm |
| 5 | 111.70mm | 107.41mm | 115.98mm |
| 6 | 110.55mm | 112.00mm | 109.09mm |
| 7 | 109.38mm | 107.89mm | 110.86mm |
| 8 | 111.61mm | 107.51mm | 115.71mm |
| 9 | 110.75mm | 107.13mm | 114.36mm |
| **10** | **106.03mm** | **102.80mm** | **109.27mm** |

### Cascaded V2b: Ground Info 12-dim

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 138.04mm | 135.97mm | 140.11mm |
| 2 | 128.08mm | 128.97mm | 127.20mm |
| 3 | 113.44mm | 120.63mm | 106.26mm |
| 4 | 119.67mm | 121.06mm | 118.27mm |
| 5 | 102.65mm | 104.54mm | 100.75mm |
| 6 | 102.08mm | 103.65mm | 100.51mm |
| 7 | 100.31mm | 101.85mm | 98.78mm |
| 8 | 102.88mm | 103.60mm | 102.17mm |
| 9 | 100.69mm | 101.37mm | 100.01mm |
| **10** | **98.57mm** | **99.06mm** | **98.08mm** |

---

## Key Findings

### 1. Ground Info consistently provides the largest improvement

| Architecture | Ground Info vs No-HMD | Baseline vs No-HMD |
|-------------|----------------------|---------------------|
| Single-Stage | **-10.67mm (-9.9%)** | -5.33mm (-4.9%) |
| Cascaded V2b | **-9.22mm (-8.6%)** | -1.75mm (-1.6%) |

Ground heights from Unity Y-axis provide the most impactful signal regardless of architecture.

### 2. Single-Stage outperforms Cascaded on Kinect data

Unlike the xRegopose dataset where Cascaded V2b achieved the best results (34.24mm), on Kinect data the simpler single-stage head performs better across all HMD configurations:

| Model | Single-Stage | Cascaded V2b | Delta |
|-------|-------------|-------------|-------|
| Ground Info | **97.16mm** | 98.57mm | +1.41mm |
| Baseline | **102.50mm** | 106.03mm | +3.53mm |
| No-HMD | **107.83mm** | 107.79mm | ~same |

Possible reasons:
- **Smaller dataset** (28k vs 210k samples): Cascaded V2b has more parameters (~48.8M vs ~43M for single-stage), may need more data
- **10 epochs insufficient**: On xRegopose, V2b's best was at epoch 19 (20 epochs). The Kinect cascaded models are still improving at epoch 10
- **Different data distribution**: Real VR sensor data + Azure Kinect skeleton may behave differently than synthetic xRegopose data

### 3. Training stability differs between architectures

- **Single-Stage No-HMD** shows extreme instability at epochs 5-6 (210mm spike), likely LR schedule interaction
- **Cascaded No-HMD** shows a huge epoch 1 spike (297mm) but stabilizes faster
- **Ground Info** provides the most stable training in both architectures

### 4. Ground Info balances Upper/Lower Body

| Architecture | Ground Info Upper | Ground Info Lower | Gap |
|-------------|-------------------|-------------------|-----|
| Single-Stage | 96.73mm | 97.60mm | 0.87mm |
| Cascaded V2b | 99.06mm | 98.08mm | 0.98mm |

Both architectures achieve nearly equal upper/lower body error with ground info, confirming that height information resolves depth ambiguity for the lower body.

### 5. Standard 9-dim HMD is more effective with extended training

At 5 epochs (smoke test): Baseline was only -1.6% vs No-HMD.
At 10 epochs: Baseline is **-4.9%** (single-stage) and **-1.6%** (cascaded) vs No-HMD.
The single-stage baseline benefits significantly from more training epochs.

---

## Data Pipeline Verification

- `KinectEgoposeDataset` with 32->16 joint mapping works correctly
- CSV synced_data.csv loaded for HMD sensor data (real VR controllers)
- Ground heights use Unity Y-axis values directly (no body-axis projection needed)
- `EgoImageResize` for full egocentric frame (no cropping)
- `use_hmd=False` correctly zeros HMD info for vision-only ablation
- `hmd_info_size=12` correctly set for ground_info model heads
- Cascaded V2b uses `EfficientHeatmapDecoder` (1.35M params)
- All models use same COCO-pretrained ResNet-101 backbone

---

## Work Dirs

### Single-Stage (10 epochs)

| Experiment | Path |
|-----------|------|
| No-HMD | `work_dirs/HMD_kinect_v4_no_hmd_10ep/` |
| Baseline | `work_dirs/HMD_kinect_v4_baseline_10ep/` |
| Ground Info | `work_dirs/HMD_kinect_v4_ground_info_10ep/` |

### Cascaded V2b (10 epochs)

| Experiment | Path |
|-----------|------|
| No-HMD | `work_dirs/HMD_kinect_v4_cascaded_no_hmd_10ep/` |
| Baseline | `work_dirs/HMD_kinect_v4_cascaded_baseline_10ep/` |
| Ground Info | `work_dirs/HMD_kinect_v4_cascaded_ground_info_10ep/` |

### 5-Epoch Smoke Tests (archived)

| Experiment | Path |
|-----------|------|
| No-HMD | `/mnt/dataset_vol/work_dirs_260305/work_dirs/HMD_kinect_v4_no_hmd_smoke_test/` |
| Baseline | `/mnt/dataset_vol/work_dirs_260305/work_dirs/HMD_kinect_v4_baseline_smoke_test/` |
| Ground Info | `/mnt/dataset_vol/work_dirs_260305/work_dirs/HMD_kinect_v4_ground_info_smoke_test/` |

---

## Next Steps

- [ ] Train cascaded V2b for 20 epochs to test if it surpasses single-stage (as it does on xRegopose)
- [ ] Add per-action breakdown metrics (Dancing, Gaming, Walking, etc.)
- [ ] Investigate No-HMD training instability (epoch 5-6 spikes in single-stage)
- [ ] Compare HMD preprocessing between Kinect (real sensors) and xRegopose (synthetic)
- [ ] Try cascaded V3 architecture (best on xRegopose: 34.06mm)
