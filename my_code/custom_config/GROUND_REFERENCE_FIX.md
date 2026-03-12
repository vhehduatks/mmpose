# Ground Reference Computation Fix for Egocentric Pose Estimation

> **Date**: 2026-02-03
> **Affects**: Mo2Cap2, xR-EgoPose, and any egocentric pose dataset using ground reference features
> **Status**: FIXED (both Mo2Cap2 and xR-EgoPose transforms updated)

---

## Problem Summary

The original `EnhanceHMDInfo_Mo2Cap2` transform computed ground reference incorrectly by assuming that one of the coordinate axes (Y-axis) corresponds to the user's height. **This assumption is WRONG** for egocentric datasets where 3D joints are in **camera coordinates**, not world coordinates.

### Why This Matters

In egocentric pose estimation:
- The camera is mounted on the user's head (HMD)
- 3D joint coordinates are relative to the camera
- **None of the X/Y/Z axes directly correspond to "up" or "down"**
- The camera orientation changes as the user moves their head

---

## Original (Incorrect) Implementation

```python
# WRONG: Assumes Y-axis is height
ground_y = min(left_foot[1], right_foot[1])
neck_from_ground = -ground_y
left_hand_from_ground = left_hand[1] - ground_y
right_hand_from_ground = right_hand[1] - ground_y
```

### Problems:
1. Uses Y-axis `[1]` which is NOT guaranteed to be vertical
2. Uses `min()` which assumes Y increases upward (also wrong)
3. Produces **meaningless values** that don't represent actual heights
4. Values can be negative when they should always be positive

---

## Correct Implementation: Body-Relative Direction

Instead of using a coordinate axis, we use the **body's own skeletal structure** to define "downward":

### Vector a (Body Downward Direction)

```python
# Pelvis center = midpoint between hip joints
pelvis_center = (L.UpLeg[11] + R.UpLeg[7]) / 2

# Vector a points from neck toward pelvis (body "downward")
vec_a = pelvis_center - neck  # neck is at origin
vec_a_unit = vec_a / np.linalg.norm(vec_a)
```

### Ground Reference Point

The ground is defined by the **farthest toe** along the body axis direction:

```python
# Project toes onto vector a
left_toe_proj = np.dot(L.ToeBase[14], vec_a_unit)
right_toe_proj = np.dot(R.ToeBase[10], vec_a_unit)

# Ground = farthest toe along body axis
ground_ref_proj = max(left_toe_proj, right_toe_proj)
```

### Ground Info Computation

```python
# Neck-to-ground distance (along body axis)
ground_head_info = ground_ref_proj  # ~1300-1500mm (realistic!)

# Hand-to-ground distance (along body axis)
left_hand_proj = np.dot(L.Hand[6], vec_a_unit)
right_hand_proj = np.dot(R.Hand[3], vec_a_unit)

left_hand_from_ground = ground_ref_proj - left_hand_proj
right_hand_from_ground = ground_ref_proj - right_hand_proj
```

---

## Visual Explanation

```
        [Neck] ←── Origin (0,0,0)
           │
           │  ← Vector a (body axis)
           ▼
      [Pelvis Center]
        /       \
   [L.UpLeg]   [R.UpLeg]
       │           │
       │           │
       ▼           ▼
   [L.Toe]     [R.Toe]
       │           │
   ════╪═══════════╪════  ← Ground Plane (perpendicular to vec_a)

   ground_head_info = distance from Neck to Ground Plane along vec_a
   hand_from_ground = distance from Hand to Ground Plane along vec_a
```

---

## Joint Indices Reference

### Mo2Cap2 (15 joints)
| Index | Joint | Usage |
|-------|-------|-------|
| 0 | Neck | Origin (root) |
| 3 | RightHand | Hand ground ref |
| 6 | LeftHand | Hand ground ref |
| 7 | RightUpLeg | Pelvis center |
| 10 | RightToeBase | Ground ref candidate |
| 11 | LeftUpLeg | Pelvis center |
| 14 | LeftToeBase | Ground ref candidate |

### xR-EgoPose (16 joints)
| Index | Joint | Usage |
|-------|-------|-------|
| 0 | Spine2 | Origin (root) |
| 4 | LeftHand | Hand ground ref |
| 7 | RightHand | Hand ground ref |
| 8 | LeftUpLeg | Pelvis center |
| 11 | LeftToeBase | Ground ref candidate |
| 12 | RightUpLeg | Pelvis center |
| 15 | RightToeBase | Ground ref candidate |

---

## Updated Files

### Mo2Cap2
- `mmpose/datasets/transforms/enhance_hmd_info_mo2cap2.py` - Fixed transform
- `my_code/visualization/mo2cap2_visualize_ground_info.py` - Visualization tool

### xR-EgoPose (FIXED 2026-02-03)
- `mmpose/datasets/transforms/enhance_hmd_info.py` - Fixed with body-axis method
- `my_code/visualization/visualize_egopose_ground_info.py` - Visualization tool

---

## Verification

### Before Fix (Incorrect Values)
```
ground_y: 671.7mm (meaningless)
neck_from_ground: -671.7mm (NEGATIVE - impossible!)
```

### After Fix (Correct Values)
```
ground_head_info: 1424.4mm (~1.4m from neck to toes - realistic!)
left_hand_from_ground: 855.2mm
right_hand_from_ground: 873.0mm
```

---

## Key Takeaways

1. **Never assume coordinate axes have physical meaning** in camera-relative systems
2. **Use body structure** (skeleton) to define directions
3. **Vector from neck to pelvis** defines the body's "downward" direction
4. **Project joints onto this vector** to get meaningful height measurements
5. This method works **regardless of camera orientation**

---

## Applying to xR-EgoPose

The same fix must be applied to `EnhanceHMDInfo` for the xR-EgoPose dataset:

```python
# xR-EgoPose joint indices
SPINE2_IDX = 0      # Root (origin)
LEFT_HAND_IDX = 4
RIGHT_HAND_IDX = 7
LEFT_UPLEG_IDX = 8
RIGHT_UPLEG_IDX = 12
LEFT_TOE_IDX = 11
RIGHT_TOE_IDX = 15

# Same algorithm:
pelvis_center = (p3d[LEFT_UPLEG_IDX] + p3d[RIGHT_UPLEG_IDX]) / 2
vec_a = pelvis_center - p3d[SPINE2_IDX]
vec_a_unit = vec_a / np.linalg.norm(vec_a)
# ... rest same as Mo2Cap2
```

---

## References

- Visualization script: `my_code/visualization/mo2cap2_visualize_ground_info.py`
- Dataset documentation: `my_code/custom_config/MO2CAP2_DATASET.md`
- Transform implementation: `mmpose/datasets/transforms/enhance_hmd_info_mo2cap2.py`
