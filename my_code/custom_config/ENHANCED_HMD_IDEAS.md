# Enhanced HMD Info Integration Ideas

## Background

The XR EgoPose dataset lacks real HMD information, so HMD info is created from GT poses. Analysis revealed that while the HMD coordinate system differs from the dataset coordinate system, **head height (Y-axis) is the same in both systems**.

This provides an opportunity to add additional global context that the current 9-dim HMD info lacks.

## Current HMD Info Structure (9 dims)

```python
hmd_info = [
    right_local,      # (3,) right hand in head-local coordinates
    left_local,       # (3,) left hand in head-local coordinates
    hand_distance,    # (1,) distance between hands
    right_distance,   # (1,) head-to-right-hand distance
    left_distance     # (1,) head-to-left-hand distance
]
```

**Limitation**: All information is relative (local coordinates, distances). No absolute position context.

---

## Enhancement Options

**IMPORTANT**: keypoint3d is ROOT-RELATIVE (head at origin), so `head_y = 0` always.
Only options that work with root-relative data are supported.

### ~~Option 1: Head Height (REMOVED)~~
- `head_y = 0` always (USELESS with root-relative data)

### Option 2: Hands Y (11 dims) ✅
```python
extra = [right_hand_y, left_hand_y]  # 9 + 2 = 11
```
- Hands Y relative to head (head is at origin)
- Tells vertical position of hands (above/below head level)

### Option 3: Torso Reference (11 dims) ✅
```python
torso_y = (right_hip_y + left_hip_y) / 2  # negative value (torso below head)
head_torso_dist = ||head - torso||
extra = [torso_y, head_torso_dist]  # 9 + 2 = 11
```
- Provides body scale information
- Connects upper/lower body context
- torso_y tells how far down the torso is from head
- **Issue**: Torso position is NOT directly measurable from real HMD

### Option 4: Ground Reference (11 dims) ✅ **RECOMMENDED FOR DEPLOYMENT**
```python
# Estimate ground from feet positions
ground_y = min(left_foot_y, right_foot_y)  # most negative (lowest foot)
head_from_ground = -ground_y  # positive value (head height from floor)
head_torso_dist = ||head - torso||  # body scale
extra = [head_from_ground, head_torso_dist]  # 9 + 2 = 11
```
- **Realistic for HMD deployment**: Real HMDs can measure height from ground
- Provides same body scale info as torso_reference
- Ground estimated from feet (caveat: assumes feet near ground)

### Option 5: Relative Heights (12 dims) ✅
```python
head_rel_y = 0 - torso_y = -torso_y  # positive (head above torso)
extra = [head_rel_y, right_hand_y - torso_y, left_hand_y - torso_y]  # 9 + 3 = 12
```
- Shifts reference from head to torso center
- All heights relative to torso

---

## Keypoint Indices (16-joint skeleton)

```python
# config.skel order:
HEAD_IDX = 0
NECK_IDX = 1
LEFT_ARM_IDX = 2
LEFT_FOREARM_IDX = 3
LEFT_HAND_IDX = 4
RIGHT_ARM_IDX = 5
RIGHT_FOREARM_IDX = 6
RIGHT_HAND_IDX = 7
LEFT_UPLEG_IDX = 8    # Left hip/pelvis
LEFT_LEG_IDX = 9
LEFT_FOOT_IDX = 10
LEFT_TOE_IDX = 11
RIGHT_UPLEG_IDX = 12  # Right hip/pelvis
RIGHT_LEG_IDX = 13
RIGHT_FOOT_IDX = 14
RIGHT_TOE_IDX = 15
```

---

## Architecture Integration Points

### Current Flow (Baseline)
```
Image → Backbone → Heatmap → Encoder(GAP) + HMD[9] → Z[64] → PoseDecoder → 3D Pose
                                    ↑
                              HMD used here only
```

### Integration Point 1: Stage 1 Encoder (Implemented)
- `Encoder.linear1(hmd_info_size, 36)` - HMD embedded in encoder
- `hmd_linear(hmd_info_size, 64)` - HMD added to latent Z
- **Status**: ✅ Implemented in `CustomxRegoposeBaselinel1_enhanced_hmd`

### Integration Point 2: Stage 2 Refinement MLP (NEW) ⭐
```
Cascaded Refinement Stage 2:
  Per-joint input (323 dims):
    ├── coarse_xyz:   3   (per-joint position)
    ├── spatial:     64   (grid-sampled backbone features)
    ├── Z:           64   (latent from Stage 1)
    ├── pose_ctx:   128   (full 48-dim pose encoded)
    └── kin_feat:    64   (15 bones × 4 dims)

  → RefinementMLP → Δpose[3] per joint
```

**Problem**: Stage 2 uses only relative features (kinematic bones, local pose). No absolute height context.

**Solution**: Add enhanced HMD to refinement input:
```
  Per-joint input (333-335 dims):
    ├── coarse_xyz:   3
    ├── spatial:     64
    ├── Z:           64
    ├── pose_ctx:   128
    ├── kin_feat:    64
    └── hmd_enhanced: 10-12  ← NEW
```

**Rationale**:
- Lower body prediction struggles with depth ambiguity
- Knowing absolute head/torso height provides vertical reference
- Kinematic features alone can't distinguish standing vs crouching

---

## Hybrid Architecture Ideas

### Idea 1: Cascaded Refinement + Attention Encoder
Replace Stage 1's standard Encoder with AttentionEncoder (from Attention Z Encoder head):
- Cross-attention preserves spatial information better than GAP
- Stage 2 refinement benefits from better coarse pose

**Concern**: Attention gate only reached 0.93% in experiments, suggesting limited benefit.

### Idea 2: Cascaded Refinement + Enhanced HMD in Stage 2 ⭐ (Recommended)
Add enhanced HMD info to Stage 2 refinement MLP input:
- Provides global context that kinematic features lack
- Minimal parameter increase (~10-12 extra input dims)
- Directly addresses lower body depth ambiguity

### Idea 3: Combined (Attention + Enhanced HMD)
Both ideas together. Worth trying if individual ideas show improvement.

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| `EnhanceHMDInfo` transform | `mmpose/datasets/transforms/enhance_hmd_info.py` | ✅ Updated (4 valid modes) |
| `CustomxRegoposeBaselinel1_enhanced_hmd` | `mmpose/models/heads/.../custom_egopose_baselinel1_head_enhanced_hmd.py` | ✅ Created |
| Stage 1 enhanced configs (4 modes) | `my_code/custom_config/HMD_xregopose_enhanced_hmd_*.py` | ✅ Created |
| `CustomEgoposeCascadedRefinementHead_enhanced` | `mmpose/models/heads/.../custom_egopose_cascaded_refinement_head_enhanced.py` | ✅ Created |
| Cascaded + Enhanced HMD config | `my_code/custom_config/HMD_xregopose_cascaded_enhanced_hmd_full_config.py` | ✅ Created |
| Cascaded + Attention Encoder | - | ❌ Not started |

---

## Experiment Results

| Mode | MPJPE | Notes |
|------|-------|-------|
| torso_reference | TBD | Shows good results (user feedback) |
| ground_reference | TBD | More realistic for HMD deployment |
| hands_y | TBD | |
| relative_heights | TBD | |

---

## Training Queue

```
# Enhanced HMD experiments (Stage 1 only - Baseline architecture)
[DONE] my_code/custom_config/HMD_xregopose_enhanced_hmd_hands_y_full_config.py
[DONE] my_code/custom_config/HMD_xregopose_enhanced_hmd_torso_ref_full_config.py
[DONE] my_code/custom_config/HMD_xregopose_enhanced_hmd_relative_heights_full_config.py
my_code/custom_config/HMD_xregopose_enhanced_hmd_ground_ref_full_config.py

# Enhanced HMD experiments (Stage 1 & Stage 2 - Cascaded Refinement)
my_code/custom_config/HMD_xregopose_cascaded_enhanced_hmd_full_config.py
```
