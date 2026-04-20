# xR-EgoPose Vision-Only Results (No HMD Input)

> Date: 2026-04-15. Fair comparison baseline — zeroed HMD input via `ZeroHMDInfo` transform.

## Summary

| Model | HMD Input | Best Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|-----------|-----------|------------|------------|
| **Cascaded + GBH** | Pseudo-HMD 12-dim | 8 | **34.06** | 25.84 | **42.29** |
| **Cascaded + LHF** | Pseudo-HMD 9-dim | 9 | 41.60 | 30.10 | 53.11 |
| Single-stage + GBH | Pseudo-HMD 12-dim | 8 | 36.28 | — | — |
| Single-stage + LHF | Pseudo-HMD 9-dim | 8 | 41.37 | 29.42 | 53.31 |
| **Single-stage (vision-only)** | **None (zero)** | **9** | **47.66** | **37.19** | **58.13** |
| **Cascaded (vision-only)** | **None (zero)** | **8** | **48.49** | **38.92** | **58.05** |

### HMD Contribution Analysis

| Comparison | MPJPE | HMD Gain |
|-----------|-------|----------|
| Single-stage: vision-only → +LHF | 47.66 → 41.37 | -6.29mm (-13.2%) |
| Single-stage: vision-only → +GBH | 47.66 → 36.28 | -11.38mm (-23.9%) |
| Cascaded: vision-only → +LHF | 48.49 → 41.60 | -6.89mm (-14.2%) |
| Cascaded: vision-only → +GBH | 48.49 → 34.06 | -14.43mm (-29.8%) |
| Pure architecture (cascaded vs single, no HMD) | 48.49 vs 47.66 | +0.83mm (cascaded slightly worse) |

### Key Findings

**1. Without HMD, cascaded does NOT help (and is slightly worse).**

Single-stage vision-only (47.66mm) outperforms cascaded vision-only (48.49mm) by 0.83mm. This confirms the finding from the Kinect dataset: Stage 2 refinement without HMD provides insufficient signal to justify its added complexity. The refinement MLP relies on HMD cues (especially ground-based heights) to make meaningful corrections.

**2. HMD provides 13-30% improvement depending on configuration.**

The pseudo-HMD features provide substantial gains:
- LHF (9-dim, hand positions): ~13-14% improvement
- GBH (12-dim, + ground heights): ~24-30% improvement
- The additional 3 ground-height dimensions provide a further 10-16% gain on top of LHF

**3. Lower body benefits most from HMD.**

| Model | Lower Body (no HMD) | Lower Body (+GBH) | Gain |
|-------|---------------------|-------------------|------|
| Single-stage | 58.13 | 42.29* | -15.84mm (-27.3%) |
| Cascaded | 58.05 | 42.29 | -15.76mm (-27.2%) |

*GBH result from Cascaded+GBH (34.06mm overall best)

**4. For paper: report vision-only as the architectural baseline.**

The vision-only results isolate the pure architectural contribution from the HMD input advantage. This addresses the fairness concern: when comparing with methods that use only images, readers can reference the vision-only rows.

---

## Per-Epoch Results

### Cascaded (vision-only) — Best: ep8, 48.49mm

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 105.77 | 73.38 | 138.16 |
| 2 | 109.72 | 86.47 | 132.98 |
| 3 | 87.12 | 79.95 | 94.29 |
| 4 | 71.99 | 61.03 | 82.96 |
| 5 | 53.13 | 36.55 | 69.70 |
| 6 | 66.76 | 45.08 | 88.44 |
| 7 | 54.36 | 46.16 | 62.57 |
| **8** | **48.49** | **38.92** | **58.05** |
| 9 | 53.92 | 43.03 | 64.81 |
| 10 | 54.37 | 45.51 | 63.23 |

### Single-stage (vision-only) — Best: ep9, 47.66mm

| Epoch | Full Body | Upper Body | Lower Body |
|-------|-----------|------------|------------|
| 1 | 141.82 | 110.77 | 172.88 |
| 2 | 125.47 | 82.57 | 168.36 |
| 3 | 103.84 | 80.27 | 127.42 |
| 4 | 75.66 | 67.25 | 84.06 |
| 5 | 56.49 | 46.18 | 66.79 |
| 6 | 74.01 | 54.78 | 93.24 |
| 7 | 53.40 | 45.56 | 61.24 |
| 8 | 47.96 | 37.63 | 58.30 |
| **9** | **47.66** | **37.19** | **58.13** |
| 10 | 60.76 | 46.92 | 74.60 |

Note: Both models show training instability (oscillation in epochs 5-7), typical for vision-only egocentric pose estimation where depth ambiguity makes optimization difficult. With HMD input, training is much more stable.

---

## Configs and Work Dirs

| Model | Config | Work Dir |
|-------|--------|----------|
| Cascaded (no HMD) | `HMD_xregopose_cascaded_no_hmd_config.py` | `work_dirs/HMD_xregopose_cascaded_no_hmd/` |
| Single-stage (no HMD) | `HMD_xregopose_single_stage_no_hmd_config.py` | `work_dirs/HMD_xregopose_single_stage_no_hmd/` |
