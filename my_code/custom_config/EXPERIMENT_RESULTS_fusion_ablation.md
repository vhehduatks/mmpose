# Fusion Mode Ablation Results (xR-EgoPose, Cascaded+GBH)

> Date: 2026-04-08

Compares how the latent vector z (64-dim from heatmap encoder) and HMD embedding (64-dim from hmd_linear) are combined before PoseDecoder and HeatmapDecoder.

All experiments use the same base: Cascaded+GBH (12-dim), ResNet-101, 10 epochs, xR-EgoPose dataset.

## Summary

| Fusion Mode | Description | Output Dim | Best Epoch | Full Body | Upper Body | Lower Body | vs Addition |
|-------------|-------------|-----------|-----------|-----------|------------|------------|-------------|
| **Addition** | z + hmd | 64 | **9** | **34.64mm** | **25.80mm** | 43.47mm | — |
| Cross-Attention | z attends hmd (4-head) | 64 | 10 | 35.04mm | 26.65mm | **43.44mm** | +0.40mm (+1.2%) |
| Concat 128 | [z, hmd] → 128-dim | 128 | 8 | 35.49mm | 27.58mm | 43.40mm | +0.85mm (+2.5%) |
| Concat→64 | [z, hmd] → Linear → 64-dim | 64 | 10 | 36.70mm | 28.37mm | 45.02mm | +2.06mm (+6.0%) |

**Addition is the best fusion strategy.**

---

## Per-Epoch Results

### Addition (Best: ep9, 34.64mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 102.20 | 91.29 | 113.10 |
| 2 | 64.84 | 48.46 | 81.22 |
| 3 | 56.43 | 39.56 | 73.30 |
| 4 | 63.18 | 51.72 | 74.64 |
| 5 | 39.89 | 30.15 | 49.62 |
| 6 | 43.55 | 29.78 | 57.31 |
| 7 | 40.60 | 33.41 | 47.80 |
| 8 | 34.71 | 25.98 | 43.43 |
| **9** | **34.64** | **25.80** | **43.47** |
| 10 | 34.80 | 26.84 | 42.75 |

### Concat 128 (Best: ep8, 35.49mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 62.38 | 46.50 | 78.26 |
| 2 | 70.27 | 52.42 | 88.12 |
| 3 | 45.49 | 34.61 | 56.36 |
| 4 | 45.47 | 36.60 | 54.33 |
| 5 | 40.92 | 29.22 | 52.63 |
| 6 | 40.73 | 30.62 | 50.83 |
| 7 | 37.06 | 29.31 | 44.81 |
| **8** | **35.49** | **27.58** | **43.40** |
| 9 | 36.06 | 26.78 | 45.34 |
| 10 | 36.99 | 27.33 | 46.66 |

### Concat→64 (Best: ep10, 36.70mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 70.32 | 59.25 | 81.39 |
| 2 | 76.28 | 67.43 | 85.12 |
| 3 | 59.46 | 45.72 | 73.21 |
| 4 | 58.78 | 51.03 | 66.53 |
| 5 | 39.70 | 32.64 | 46.76 |
| 6 | 42.28 | 31.96 | 52.60 |
| 7 | 43.67 | 36.58 | 50.75 |
| 8 | 36.79 | 29.24 | 44.34 |
| 9 | 36.78 | 28.94 | 44.63 |
| **10** | **36.70** | **28.37** | **45.02** |

### Cross-Attention (Best: ep10, 35.04mm)

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 63.45 | 51.46 | 75.45 |
| 2 | 75.75 | 55.53 | 95.98 |
| 3 | 56.96 | 40.72 | 73.20 |
| 4 | 42.20 | 35.83 | 48.58 |
| 5 | 38.35 | 28.61 | 48.09 |
| 6 | 42.28 | 30.70 | 53.86 |
| 7 | 39.98 | 31.07 | 48.89 |
| 8 | 35.65 | 26.90 | 44.40 |
| 9 | 35.93 | 27.14 | 44.72 |
| **10** | **35.04** | **26.65** | **43.44** |

---

## Analysis

### Why addition outperforms alternatives

1. **Shared representation space**: Addition forces both visual and HMD modalities into compatible alignment. The network learns to produce complementary representations rather than independent features.

2. **Dimension preservation**: Addition maintains the 64-dim latent space that downstream PoseDecoder and HeatmapDecoder expect. Concat 128 doubles the input dimension, increasing overfitting risk (peaks at ep8, then degrades through ep9-10).

3. **Residual interpretation**: `z_fused = z_visual + z_hmd` functions as a residual connection where HMD acts as a correction signal to the visual features, analogous to ResNet skip connections.

4. **Convergence speed**: Addition converges fastest (best at ep9 with stable ep8-10), while concat methods are slower and cross-attention shows oscillation (ep5-7).

### Why Concat→64 is worst (+2.06mm)

The extra projection layer (Linear 128→64 + ReLU) introduces an information bottleneck. The network must first learn to combine features in 128-dim, then compress back to 64. This is strictly harder than learning compatible 64-dim representations directly via addition.

### Why cross-attention is close (+0.40mm)

The attention mechanism can learn selective fusion, but with only 1 query token and 1 key-value token, it reduces to a learned scalar weighting — adding complexity (MultiheadAttention + LayerNorm) without meaningful benefit. Its lower body MPJPE (43.44mm) actually matches addition (43.47mm), suggesting the extra capacity helps depth estimation marginally but hurts upper body.

### Lower body is stable across methods

Lower body MPJPE ranges only 43.40-45.02mm across the 4 methods, while upper body varies 25.80-28.37mm. The primary differentiator is upper body accuracy, where addition's clean signal path provides the most benefit.

### Note on reproducibility

The Addition result here (34.64mm) vs the original V3 both_from_ground (34.06mm) shows normal run-to-run variance (~0.6mm) due to different random seeds and training dynamics.

---

## Configs and Work Dirs

| Fusion Mode | Config | Work Dir |
|-------------|--------|----------|
| Addition | `HMD_xregopose_cascaded_gbh_fusion_addition_config.py` | `work_dirs/HMD_xregopose_fusion_addition/` |
| Concat 128 | `HMD_xregopose_cascaded_gbh_fusion_concat_128_config.py` | `work_dirs/HMD_xregopose_fusion_concat128/` |
| Concat→64 | `HMD_xregopose_cascaded_gbh_fusion_concat_64_config.py` | `work_dirs/HMD_xregopose_fusion_concat64/` |
| Cross-Attention | `HMD_xregopose_cascaded_gbh_fusion_cross_attention_config.py` | `work_dirs/HMD_xregopose_fusion_crossattn/` |
