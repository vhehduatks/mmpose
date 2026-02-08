# Mo2Cap2 Cascaded vs Baseline Performance Analysis

## Summary

Surprisingly, on Mo2Cap2 dataset, the **Baseline model (82.08mm) slightly outperforms Cascaded (82.28mm)** with ground reference and matched LR (0.00025). This document analyzes potential causes and proposes solutions.

## Results Comparison

| Model | Full Body | Upper Body | Lower Body | Best Epoch |
|-------|-----------|------------|------------|------------|
| Baseline + Ground (lr=0.00025) | **82.08mm** | 79.44mm | **84.39mm** | 7 |
| Cascaded + Ground (lr=0.00025) | 82.28mm | **78.39mm** | 85.68mm | 9 |

---

## Hypothesis 1: Loss Weight Imbalance

### Problem

Both stages have equal loss weight, causing gradient competition:

```python
loss_pose_l2norm = 1.0        # Coarse pose (Stage 1)
loss_pose_l2norm_refined = 1.0  # Refined pose (Stage 2)
```

### Why This Matters

- Stage 1 output is used as input to Stage 2
- Equal weights mean refinement might learn to "undo" Stage 1's work
- Gradients from Stage 2 losses backpropagate to Stage 1, potentially destabilizing learning

### Solution: Asymmetric Loss Weights

```python
loss_pose_l2norm = 0.3        # Lower weight for coarse
loss_pose_l2norm_refined = 1.0  # Full weight for refined
```

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v3_config.py`

---

## Hypothesis 2: Gradient Flow Competition

### Problem

Stage 2 losses backpropagate through `coarse_pose` to Stage 1:
- `loss_pose_l2norm_refined` (1.0)
- `loss_bone_length` (0.5)
- `loss_symmetry` (0.1)

This creates competing gradient signals in Stage 1.

### Solution: Detach Coarse Pose

```python
# In refine() method:
coarse_pose = coarse_pose.detach()  # Stop gradients to Stage 1
```

**Status**: Requires code modification to head file.

---

## Hypothesis 3: Learning Rate Mismatch

### Problem

Cascaded model has ~1M more parameters (RefinementMLP), but uses the same LR as Baseline.

### Evidence: Convergence Behavior

| Epoch | Baseline | Cascaded |
|-------|----------|----------|
| 1 | 100.67mm | 91.44mm |
| 2 | 84.17mm | 93.09mm ⚠️ (worse) |
| 7 | **82.08mm** | 82.49mm |
| 9 | 82.76mm | **82.28mm** |

Cascaded starts better but oscillates, suggesting unstable optimization.

### Solution: Layer-wise LR

```python
optim_wrapper = dict(
    optimizer=dict(lr=0.00025, type='AdamW'),
    paramwise_cfg=dict(
        custom_keys={
            'refinement_mlp': dict(lr_mult=2.0),  # Higher LR for refinement
            'encoder': dict(lr_mult=0.5),         # Lower LR for Stage 1
        }
    )
)
```

---

## Hypothesis 4: Mo2Cap2 Skeleton Specifics

### Problem

Mo2Cap2 has a different skeleton structure than xR-EgoPose:

| Factor | xR-EgoPose | Mo2Cap2 |
|--------|------------|---------|
| Joints | 16 | 15 |
| Root joint | Pelvis | Neck |
| Skeleton | Full body with pelvis | No pelvis |

The kinematic features computed in Stage 2 may be less informative for Mo2Cap2's neck-rooted skeleton.

### Solution

Modify kinematic feature computation to better suit Mo2Cap2's structure, or add pelvis-relative features.

---

## Hypothesis 5: Training Epochs

### Observation

- Baseline: Best at epoch 7
- Cascaded: Best at epoch 9

Cascaded may need more epochs due to more parameters.

### Solution

Extend training to 15-20 epochs with milestones [10, 14].

---

## Experiment Plan

### Priority 1: Asymmetric Loss Weights (Testing)

**Config**: `HMD_mo2cap2_cascaded_both_from_ground_v3_config.py`

Changes:
- `loss_pose_l2norm`: 1.0 → 0.3
- Keep `loss_pose_l2norm_refined`: 1.0

Expected outcome: Better Stage 2 learning without destabilizing Stage 1.

### Priority 2: Gradient Detachment

Modify `custom_mo2cap2_cascaded_refinement_head_enhanced.py`:
```python
def loss(...):
    # Detach coarse pose for refinement
    refined_pose = self.refine(
        coarse_pose.detach(),  # <-- Add .detach()
        pred_fields, backbone_feat, z,
        hmd_info=HMD_info
    )
```

### Priority 3: Layer-wise LR

Test with different LR multipliers for Stage 1 vs Stage 2.

---

## xR-EgoPose Comparison

On xR-EgoPose, cascaded refinement **does** help:

| Model | xR-EgoPose MPJPE |
|-------|------------------|
| Baseline | ~41.37mm |
| Cascaded | ~41.60mm |
| Cascaded + Ground V3 | **34.06mm** |

The ground reference is the dominant factor (-7mm improvement), and cascaded refinement provides additional benefit when combined with it.

On Mo2Cap2, the ground reference also provides ~6-7mm improvement, but cascaded refinement doesn't add value (and slightly hurts performance).

---

## Conclusion

The most likely cause is **Hypothesis 1 (Loss Weight Imbalance)**. The equal weights cause Stage 1 and Stage 2 to compete for gradients, preventing optimal learning of either stage.

Testing asymmetric loss weights should reveal whether this is the root cause.
