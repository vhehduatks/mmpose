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
| **Cascaded V3b (Asym + LR=0.0005)** | 12-dim | **81.14mm** 🏆 | **76.49mm** | 85.21mm | **NEW BEST** |
| Cascaded V3 (Asym + LR=0.00025) | 12-dim | 81.89mm | 79.68mm | **83.83mm** | Previous best |
| Baseline + Ground (lr=0.00025) | 12-dim | 82.08mm | 79.44mm | 84.39mm | |
| Cascaded V2 (lr=0.00025) | 12-dim | 82.28mm | 78.39mm | 85.68mm | Equal loss weights |
| Cascaded V3a (Asym + LR=0.000125) | 12-dim | 84.34mm | 82.95mm | 85.56mm | Lower LR |
| Baseline + Ground (lr=0.0005) | 12-dim | 85.30mm | 87.42mm | 83.44mm | |
| Cascaded No Ground | 9-dim | 88.31mm | 89.03mm | 87.67mm | No ground ref |
| Baseline No Ground | 9-dim | 92.59mm | 93.01mm | 92.22mm | No ground ref |
| HEAD from ground | 10-dim | 148.88mm | 145.88mm | 151.50mm | Head height only |
| HAND from ground | 11-dim | 171.53mm | 163.12mm | 178.89mm | Hand heights only |

---

## Detailed Results

### 1. Cascaded BOTH From Ground V3b (Asymmetric Loss + Higher LR) - NEW BEST 🏆

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v3b_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 12-dim (9 base + 3 ground) |
| **Learning Rate** | **0.0005** (doubled from V3) |
| **loss_pose_l2norm** | **0.3** (asymmetric) |
| loss_pose_l2norm_refined | 1.0 |
| Epochs | 10 |
| Best Epoch | 4 |

#### Overall Results (Best Epoch 4)

| Metric | MPJPE (mm) |
|--------|------------|
| **Full Body** | **81.14** 🏆 |
| **Upper Body** | **76.49** |
| Lower Body | 85.21 |

#### Key Insight: Higher LR + Asymmetric Loss

With asymmetric loss weights properly balancing gradient flow, higher learning rate enables faster and better convergence:

| Change | V3 → V3b |
|--------|----------|
| Learning Rate | 0.00025 → **0.0005** |
| Full Body | 81.89 → **81.14mm** (-0.75mm) |
| Upper Body | 79.68 → **76.49mm** (-3.19mm) |
| Best Epoch | 7 → **4** (faster convergence) |

---

### 2. Cascaded BOTH From Ground V3 (Asymmetric Loss + LR=0.00025)

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v3_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 12-dim (9 base + 3 ground) |
| Learning Rate | 0.00025 |
| **loss_pose_l2norm** | **0.3** (reduced from 1.0) |
| loss_pose_l2norm_refined | 1.0 |
| Epochs | 10 |
| Best Epoch | 7 |

#### Overall Results (Best Epoch 7)

| Metric | MPJPE (mm) |
|--------|------------|
| **Full Body** | **81.89** |
| Upper Body | 79.68 |
| **Lower Body** | **83.83** (best lower body) |

#### Per-Action Breakdown

| Action | MPJPE (mm) | Difficulty |
|--------|------------|------------|
| walking | 69.76 | Easy |
| waving | 71.72 | Easy |
| boxing | 71.94 | Easy |
| dancing | 72.78 | Medium |
| crawling | 95.48 | Hard |
| sitting | 96.39 | Hard |
| crouching | 98.36 | Hard |
| stretching | 105.34 | Very Hard |

#### Epoch Progression

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 113.12 | 115.08 | 111.41 |
| 2 | 91.67 | 89.29 | 93.75 |
| 3 | 88.54 | 85.10 | 91.55 |
| 4 | 82.48 | 79.36 | 85.20 |
| 5 | 130.09 ⚠️ | 115.71 | 142.67 |
| 6 | 87.47 | 85.93 | 88.83 |
| **7** | **81.89** | **79.68** | **83.83** |
| 8 | 83.47 | 81.46 | 85.24 |
| 9 | 82.23 | 80.58 | 83.67 |
| 10 | 83.34 | 81.94 | 84.56 |

#### Key Insight: Asymmetric Loss Weights

The original cascaded model (V2) had equal loss weights for coarse and refined poses, causing gradient competition. By reducing the coarse pose loss weight (`loss_pose_l2norm: 1.0 → 0.3`), the refinement stage can dominate optimization.

| Change | V2 → V3 |
|--------|---------|
| Full Body | 82.28 → **81.89mm** (-0.39mm) |
| Lower Body | 85.68 → **83.83mm** (-1.85mm) |

---

### 3. Cascaded BOTH From Ground V3a (Asymmetric Loss + Lower LR)

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v3a_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 12-dim (9 base + 3 ground) |
| **Learning Rate** | **0.000125** (halved from V3) |
| loss_pose_l2norm | 0.3 (asymmetric) |
| loss_pose_l2norm_refined | 1.0 |
| Epochs | 10 |
| Best Epoch | 9 |

#### Overall Results (Best Epoch 9)

| Metric | MPJPE (mm) |
|--------|------------|
| Full Body | 84.34 |
| Upper Body | 82.95 |
| Lower Body | 85.56 |

#### Key Insight: Lower LR Underperforms

Lower learning rate with asymmetric loss leads to underperformance:
- Slower convergence (best at epoch 9)
- Worse overall results (+2.45mm vs V3)

---

### 4. Cascaded BOTH From Ground V2 (lr=0.00025)

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v2_config.py`

| Setting | Value |
|---------|-------|
| Architecture | Cascaded Refinement |
| HMD Info | 12-dim (9 base + 3 ground: head + left_hand + right_hand) |
| Learning Rate | 0.00025 |
| loss_pose_l2norm | 1.0 (equal weights) |
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

### 5. HEAD From Ground (Body-Axis)

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

### 6. HAND From Ground (Body-Axis)

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

Learning rate effect depends on loss weight configuration:

**Baseline + Ground (equal loss weights)**:
- LR 0.00025: 82.08mm ✅ (lower LR is better)
- LR 0.0005: 85.30mm

**Cascaded + Asymmetric Loss (0.3/1.0)**:
- LR 0.0005: **81.14mm** 🏆 (higher LR is better!)
- LR 0.00025: 81.89mm
- LR 0.000125: 84.34mm

**Insight**: Asymmetric loss weights stabilize training, allowing higher LR to be beneficial.

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

### Loss Functions (V3b Best Configuration)

| Loss | Weight | Purpose |
|------|--------|---------|
| loss_kpt (heatmap MSE) | 1000 | Primary 2D heatmap |
| loss_heatmap_recon | 500 | Heatmap reconstruction |
| **loss_pose_l2norm** | **0.3** | Coarse 3D pose (asymmetric) |
| loss_cosine_similarity | 0.1 | Bone direction |
| loss_limb_length | 0.25 | L1 distance |
| loss_hmd | 1.0 | HMD reconstruction |
| **loss_pose_l2norm_refined** | **1.0** | Refined 3D pose (full weight) |
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
| Mo2Cap2 | V3b Asym + Higher LR | **81.14mm** |

Mo2Cap2 is more challenging due to:
- More diverse action categories (8 vs xR-EgoPose)
- Different camera viewpoints and environments
- Official MATLAB evaluation protocol (skeleton rescaling + Procrustes)

---

## Ablation Study Results (2026-02-07)

### Complete Results Table

| # | Config | Arch | Ground | LR | Loss Weight | Best Epoch | Full Body | Upper Body | Lower Body |
|---|--------|------|--------|-----|-------------|------------|-----------|------------|------------|
| 1 | baseline_no_ground | Baseline | 9-dim | 0.0005 | - | 7 | 92.59mm | 93.01mm | 92.22mm |
| 2 | baseline | Baseline | 12-dim | 0.0005 | - | 2 | 85.30mm | 87.42mm | 83.44mm |
| 3 | baseline_with_ground_lr | Baseline | 12-dim | 0.00025 | - | 7 | 82.08mm | 79.44mm | 84.39mm |
| 4 | cascaded_no_ground | Cascaded | 9-dim | 0.00025 | Equal (1.0) | 2 | 88.31mm | 89.03mm | 87.67mm |
| 5 | cascaded_both_v2 | Cascaded | 12-dim | 0.00025 | Equal (1.0) | 9 | 82.28mm | 78.39mm | 85.68mm |
| 6 | cascaded_both_v3 | Cascaded | 12-dim | 0.00025 | Asym (0.3) | 7 | 81.89mm | 79.68mm | **83.83mm** |
| 7 | cascaded_both_v3a | Cascaded | 12-dim | 0.000125 | Asym (0.3) | 9 | 84.34mm | 82.95mm | 85.56mm |
| 8 | **cascaded_both_v3b** | Cascaded | 12-dim | **0.0005** | **Asym (0.3)** | 4 | **81.14mm** 🏆 | **76.49mm** | 85.21mm |

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

| Comparison | Baseline | Cascaded V2 | Cascaded V3 | Best |
|------------|----------|-------------|-------------|------|
| Without Ground (9-dim) | 92.59mm | 88.31mm | - | Cascaded |
| With Ground, Equal Loss | 82.08mm | 82.28mm | - | Baseline |
| With Ground, Asym Loss | 82.08mm | - | **81.89mm** | **Cascaded V3** |

**🎯 Key Finding**: Cascaded refinement **does help** when loss weights are properly balanced!
- V2 (equal weights): Baseline wins by 0.20mm
- V3 (asymmetric weights): **Cascaded wins by 0.19mm**

#### 4. Loss Weight Effect (Cascaded Models)

| Loss Weight | Full Body | Upper Body | Lower Body | Δ Full |
|-------------|-----------|------------|------------|--------|
| Equal (1.0/1.0) | 82.28mm | **78.39mm** | 85.68mm | baseline |
| **Asymmetric (0.3/1.0)** | **81.89mm** | 79.68mm | **83.83mm** | **-0.39mm** |

**Lower coarse loss weight enables better refinement learning**, especially for lower body (-1.85mm).

#### 5. Learning Rate Effect with Asymmetric Loss (V3 Ablation)

| LR | Full Body | Upper Body | Lower Body | Best Epoch | Δ Full |
|----|-----------|------------|------------|------------|--------|
| 0.000125 (V3a) | 84.34mm | 82.95mm | 85.56mm | 9 | +2.45mm (worse) |
| 0.00025 (V3) | 81.89mm | 79.68mm | **83.83mm** | 7 | baseline |
| **0.0005 (V3b)** | **81.14mm** | **76.49mm** | 85.21mm | **4** | **-0.75mm (best)** |

**🎯 Key Finding**: Higher LR (0.0005) works **better** with asymmetric loss!
- Faster convergence (best at epoch 4 vs 7)
- Significantly better upper body (-3.19mm)
- Lower body slightly worse (+1.38mm) - tradeoff for overall improvement

#### 6. Summary: What Matters Most

| Factor | Impact | Notes |
|--------|--------|-------|
| **Ground Reference** | **-6~7mm** | Most important factor |
| **LR + Asymmetric Loss** | **-0.9mm** | Higher LR (0.0005) with asymmetric loss |
| **Loss Weight Balance** | **-0.4mm** | Asymmetric (0.3/1.0) for cascaded |
| **Architecture** | **-0.2mm** | Cascaded wins with proper loss weights |

### Best Configuration

**Cascaded V3b + Ground Reference + Asymmetric Loss (0.3/1.0) + Higher LR (0.0005)** achieves **81.14mm** 🏆

| Rank | Model | Full Body | Upper Body | Lower Body |
|------|-------|-----------|------------|------------|
| 1 | **Cascaded V3b (Asym + LR=0.0005)** | **81.14mm** | **76.49mm** | 85.21mm |
| 2 | Cascaded V3 (Asym + LR=0.00025) | 81.89mm | 79.68mm | **83.83mm** |
| 3 | Baseline + Ground (LR=0.00025) | 82.08mm | 79.44mm | 84.39mm |
| 4 | Cascaded V2 (Equal) | 82.28mm | 78.39mm | 85.68mm |
| 5 | Cascaded V3a (Asym + LR=0.000125) | 84.34mm | 82.95mm | 85.56mm |

**Note**: V3b achieves best overall and upper body, while V3 has best lower body.

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
- [x] Ablation study on loss weights ✅
  - Created `HMD_mo2cap2_cascaded_both_from_ground_v3_config.py`
  - Asymmetric loss (0.3/1.0) achieves 81.89mm
  - See `MO2CAP2_CASCADED_ANALYSIS.md` for detailed analysis
- [x] LR ablation with asymmetric loss (V3a, V3b) ✅
  - V3a (lr=0.000125): 84.34mm (worse)
  - V3 (lr=0.00025): 81.89mm
  - **V3b (lr=0.0005): 81.14mm 🏆 (NEW BEST)**
- [ ] Per-joint MPJPE analysis
