# Mo2Cap2 Experiment Results

This document tracks experiment results for the Mo2Cap2 egocentric pose estimation dataset.

## Dataset Information

| Property | Value |
|----------|-------|
| Training Samples | ~530,000 (H5 chunks) |
| Test Samples | 5,646 |
| Joints | 15 |
| Coordinate Unit | Meters (converted from mm) |
| Evaluation Protocol | Official MATLAB (skeleton rescaling + Procrustes) |

### Test Set Environments

| Environment | Sequence | Samples | Description |
|-------------|----------|---------|-------------|
| **Outdoor** | olek_outdoor | 2,744 | Outdoor environment with natural lighting |
| **Indoor** | weipeng_studio | 2,902 | Indoor studio environment |

### Action Categories

| Action | Description |
|--------|-------------|
| walking | Normal walking |
| sitting | Sitting poses |
| crawling | Crawling on ground |
| crouching | Crouching poses |
| boxing | Boxing movements |
| dancing | Dance movements |
| stretching | Stretching exercises |
| waving | Waving gestures |

---

## Model Comparison Summary

| Model | HMD Info | Full Body | Upper Body | Lower Body | Notes |
|-------|----------|-----------|------------|------------|-------|
| **Baseline + Ground (lr=0.00025)** | 12-dim | **82.08mm** 🏆 | 79.44mm | 84.39mm | **NEW BEST** |
| Cascaded V2 (lr=0.00025) | 12-dim | 82.28mm | **78.39mm** | 85.68mm | Previous best |
| Baseline + Ground (lr=0.0005) | 12-dim | 85.30mm | 87.42mm | **83.44mm** | Higher LR |
| Cascaded No Ground | 9-dim | 88.31mm | 89.03mm | 87.67mm | No ground ref |
| Baseline No Ground | 9-dim | 92.59mm | 93.01mm | 92.22mm | No ground ref |
| HEAD from ground | 10-dim | 148.88mm | 145.88mm | 151.50mm | Head height only |
| HAND from ground | 11-dim | 171.53mm | 163.12mm | 178.89mm | Hand heights only |

---

## Detailed Results

### 1. Cascaded BOTH From Ground V2 (lr=0.00025) - BEST

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v2_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 12-dim (9 base + 3 ground: head + left_hand + right_hand) |
| Learning Rate | 0.00025 |
| Epochs | 10 |
| Best Epoch | 9 |

#### Overall Results (Best Epoch 9)

| Metric | MPJPE (mm) |
|--------|------------|
| **Full Body** | **82.28** |
| Upper Body | 78.39 |
| Lower Body | 85.68 |

#### Per-Action Breakdown

| Action | MPJPE (mm) | Difficulty |
|--------|------------|------------|
| walking | 67.88 | Easy |
| waving | 68.95 | Easy |
| dancing | 73.60 | Medium |
| boxing | 76.47 | Medium |
| crawling | 96.47 | Hard |
| crouching | 96.88 | Hard |
| sitting | 96.94 | Hard |
| stretching | 112.21 | Very Hard |

#### Epoch Progression

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 91.44 | 88.89 | 93.68 |
| 2 | 93.09 | 90.63 | 95.24 |
| 3 | 92.45 | 86.75 | 97.44 |
| 4 | 85.32 | 81.49 | 88.66 |
| 5 | 86.80 | 82.96 | 90.16 |
| 6 | 82.93 | 80.52 | 85.04 |
| 7 | 82.49 | 78.13 | 86.31 |
| 8 | 83.16 | 79.15 | 86.67 |
| **9** | **82.28** | **78.39** | **85.68** |
| 10 | 84.11 | 80.51 | 87.26 |

---

### 2. HEAD From Ground (Body-Axis)

**Config**: `HMD_mo2cap2_cascaded_head_from_ground_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 10-dim (9 base + 1 head height) |
| Learning Rate | 0.0005 |
| Best Epoch | 4 |

#### Results

| Metric | MPJPE (mm) |
|--------|------------|
| Full Body | 148.88 |
| Upper Body | 145.88 |
| Lower Body | 151.50 |

---

### 3. HAND From Ground (Body-Axis)

**Config**: `HMD_mo2cap2_cascaded_hand_from_ground_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 11-dim (9 base + 2 hand heights) |
| Learning Rate | 0.0005 |

#### Results

| Metric | MPJPE (mm) |
|--------|------------|
| Full Body | 171.53 |
| Upper Body | 163.12 |
| Lower Body | 178.89 |

---

## Key Findings

### 1. Ground Reference Effectiveness

The "both_from_ground" mode (head + hand heights) significantly outperforms single-dimension variants:

| Ground Info | Full Body MPJPE | Improvement vs HEAD |
|-------------|-----------------|---------------------|
| HEAD only (10-dim) | 148.88mm | baseline |
| HAND only (11-dim) | 171.53mm | -22.65mm (worse) |
| **BOTH (12-dim)** | **82.28mm** | **+66.60mm (-44.7%)** |

### 2. Learning Rate Impact

Lower learning rate (0.00025 vs 0.0005) provided more stable training:
- V2 (lr=0.00025): Best at epoch 9 with 82.28mm
- Smoother convergence curve
- Less overfitting at later epochs

### 3. Action Difficulty Analysis

Actions can be categorized by difficulty:
- **Easy** (<75mm): walking, waving
- **Medium** (75-90mm): dancing, boxing
- **Hard** (90-100mm): crawling, crouching, sitting
- **Very Hard** (>100mm): stretching

### 4. Body Part Analysis

- Upper body (78.39mm) performs better than lower body (85.68mm)
- Ground reference heights help both parts significantly
- Hand heights provide crucial depth cues for arm/leg estimation

---

## Training Configuration

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| loss_kpt (heatmap MSE) | 1000 | Primary 2D heatmap |
| loss_heatmap_recon | 500 | Heatmap reconstruction |
| loss_pose_l2norm | 1.0 | Coarse 3D pose |
| loss_cosine_similarity | 0.1 | Bone direction |
| loss_limb_length | 0.25 | L1 distance |
| loss_hmd | 1.0 | HMD reconstruction |
| loss_pose_l2norm_refined | 1.0 | Refined 3D pose |
| loss_bone_length | 0.5 | Per-bone length |
| loss_symmetry | 0.1 | Left-right symmetry |

### LR Schedule

```
MultiStepLR:
  milestones: [6, 8]
  gamma: 0.1
```

---

## Comparison with xR-EgoPose

| Dataset | Model | Full Body MPJPE |
|---------|-------|-----------------|
| xR-EgoPose | V3 Both From Ground | **34.06mm** |
| Mo2Cap2 | BOTH V2 (lr=0.00025) | **82.28mm** |

Mo2Cap2 is more challenging due to:
- More diverse action categories (8 vs xR-EgoPose)
- Different camera viewpoints and environments
- Official MATLAB evaluation protocol (skeleton rescaling + Procrustes)

---

## Ablation Study Results (2026-02-07)

### Complete Results Table

| # | Config | Arch | Ground | LR | Best Epoch | Full Body | Upper Body | Lower Body |
|---|--------|------|--------|-----|------------|-----------|------------|------------|
| 1 | baseline_no_ground | Baseline | 9-dim | 0.0005 | 7 | 92.59mm | 93.01mm | 92.22mm |
| 2 | baseline | Baseline | 12-dim | 0.0005 | 2 | 85.30mm | 87.42mm | 83.44mm |
| 3 | baseline_with_ground_lr | Baseline | 12-dim | 0.00025 | 7 | **82.08mm** | **79.44mm** | 84.39mm |
| 4 | cascaded_no_ground | Cascaded | 9-dim | 0.00025 | 2 | 88.31mm | 89.03mm | 87.67mm |
| 5 | cascaded_both_v2 | Cascaded | 12-dim | 0.00025 | 9 | 82.28mm | 78.39mm | **85.68mm** |

### Key Findings

#### 1. Ground Reference Effect

| Architecture | Without Ground | With Ground | Improvement |
|--------------|----------------|-------------|-------------|
| Baseline | 92.59mm | 85.30mm | **-7.29mm (-7.9%)** |
| Cascaded | 88.31mm | 82.28mm | **-6.03mm (-6.8%)** |

**Ground reference provides consistent 6-7mm improvement across architectures.**

#### 2. Learning Rate Effect (Baseline + Ground)

| LR | Full Body | Upper Body | Lower Body | Δ Full |
|----|-----------|------------|------------|--------|
| 0.0005 | 85.30mm | 87.42mm | 83.44mm | baseline |
| 0.00025 | 82.08mm | 79.44mm | 84.39mm | **-3.22mm (-3.8%)** |

**Lower LR provides 3.22mm improvement** with better upper body performance.

#### 3. Architecture Effect (Matched Conditions)

| Comparison | Baseline | Cascaded | Δ |
|------------|----------|----------|---|
| Without Ground (9-dim) | 92.59mm | 88.31mm | **-4.28mm (Cascaded wins)** |
| With Ground, LR=0.00025 | **82.08mm** | 82.28mm | **+0.20mm (Baseline wins!)** |

**🎯 Surprising Finding**: With ground reference and matched LR, **Baseline slightly outperforms Cascaded!**
- Baseline (82.08mm) vs Cascaded (82.28mm) = **0.20mm better**
- Upper Body: Baseline (79.44mm) vs Cascaded (78.39mm) = Cascaded 1.05mm better
- Lower Body: Baseline (84.39mm) vs Cascaded (85.68mm) = Baseline 1.29mm better

#### 4. Summary: What Matters Most

| Factor | Impact | Notes |
|--------|--------|-------|
| **Ground Reference** | **-6~7mm** | Most important factor |
| **Learning Rate** | **-3mm** | Lower LR (0.00025) is better |
| **Architecture** | **±0.2mm** | Minimal difference with ground ref |

### Best Configuration

**Baseline + Ground Reference + LR=0.00025** achieves **82.08mm** (slightly better than Cascaded V2's 82.28mm)

---

## Environment Breakdown

### Cascaded BOTH V2 (lr=0.00025) - Best Epoch 9

| Environment | Samples | Full Body | Upper Body | Lower Body |
|-------------|---------|-----------|------------|------------|
| **Overall** | 5,646 | 82.29mm | 78.34mm | 85.75mm |
| Outdoor (olek) | 2,744 | 85.26mm | 73.84mm | 95.26mm |
| Indoor (weipeng) | 2,902 | 79.47mm | 82.59mm | 76.75mm |

### Key Observations

1. **Indoor performs better overall**: 79.47mm vs 85.26mm (-5.79mm)
   - Indoor studio has controlled lighting and simpler backgrounds

2. **Upper Body is better outdoors**: 73.84mm vs 82.59mm (-8.75mm)
   - Outdoor natural lighting may provide better contrast for upper body

3. **Lower Body is dramatically better indoors**: 76.75mm vs 95.26mm (-18.51mm)
   - Indoor environment has consistent floor/background
   - Outdoor ground variations cause more depth ambiguity

4. **Environment affects body parts differently**:
   - Outdoor: Upper body 73.84mm (good) vs Lower body 95.26mm (poor) → 21.42mm gap
   - Indoor: Upper body 82.59mm vs Lower body 76.75mm → 5.84mm gap (more balanced)

---

## TODO

- [x] Add baseline model (no cascaded refinement) for comparison
  - Created `CustomMo2Cap2BaselineHead` and `HMD_mo2cap2_baseline_config.py`
- [x] Per-environment (indoor/outdoor) evaluation
  - Modified `mo2cap2_evaluate.py` and `custom_mo2cap2_metric.py`
  - Now reports outdoor (olek_outdoor) and indoor (weipeng_studio) MPJPE separately
- [x] Create ablation study configs
  - Ground reference: with/without
  - Learning rate: 0.0005 vs 0.00025
  - Architecture: Baseline vs Cascaded
- [x] Run ablation experiments ✅
- [ ] Per-joint MPJPE analysis
- [ ] Ablation study on loss weights
