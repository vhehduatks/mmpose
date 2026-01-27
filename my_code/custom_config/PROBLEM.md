# Code Fixes Required

> Last updated: 2026-01-21

---

## 1. Metric squeeze() Bug (Critical)

**File**: `mmpose/evaluation/metrics/custom_xr_egopose_metric.py`
**Location**: line 214-215

### Current Code
```python
pred_batch_3d_keypoints = torch.stack(pred_batch_3d_keypoints).squeeze()
gt_batch_keypoint_3d = torch.stack(gt_batch_keypoint_3d).squeeze()
```

### Problem
`squeeze()` removes all size-1 dimensions, causing the batch dimension to disappear when N=1.

| Samples | After stack | After squeeze | Expected | Result |
|---------|----------|------------|------|------|
| N > 1 | [N, 1, 16, 3] | [N, 16, 3] | OK | ✅ |
| **N = 1** | [1, 1, 16, 3] | **[16, 3]** | [1, 16, 3] | ❌ |

### Impact
- When N=1, 16 joints are misinterpreted as 16 samples
- MPJPE calculation is completely wrong

### Fix
```python
# Use squeeze(dim=1) to remove only the instance dimension
pred_batch_3d_keypoints = torch.stack(pred_batch_3d_keypoints).squeeze(dim=1)
gt_batch_keypoint_3d = torch.stack(gt_batch_keypoint_3d).squeeze(dim=1)
```

### Priority
**Critical** - Always occurs during single-sample testing

---

## 2. Fragile Transpose Logic

**File**: `mmpose/evaluation/metrics/mo2cap2_evaluate.py`
**Location**: `compute_error()` function, line 616-621

### Current Code
```python
if pred.shape[1] != 3:
    pred = np.transpose(pred, [1, 0])

if gt.shape[1] != 3:
    gt = np.transpose(gt, [1, 0])
```

### Problem
- Assumes only 2D arrays `[16, 3]` or `[3, 16]`
- Unexpected behavior with 3D array input
- Implicit conversion without shape validation

### Fix
```python
# Add explicit shape validation
assert pred.ndim == 2, f"Expected 2D array, got {pred.ndim}D with shape {pred.shape}"
assert pred.shape == (16, 3) or pred.shape == (3, 16), f"Unexpected shape: {pred.shape}"

if pred.shape[1] != 3:
    pred = np.transpose(pred, [1, 0])
```

### Priority
**Medium** - Only occurs with non-standard input

---

## 3. Missing Shape Validation

**File**: `mmpose/evaluation/metrics/custom_xr_egopose_metric.py`
**Location**: `process()` and `compute_metrics()`

### Problem
- No validation on input keypoint3d shape
- Silent error with incorrect shape input

### Fix
```python
# Add validation in process()
pred['keypoint3d'] = data_sample['pred_instances']['keypoint_3d']
assert pred['keypoint3d'].shape[-2:] == (16, 3), \
    f"Expected keypoint3d shape [..., 16, 3], got {pred['keypoint3d'].shape}"
```

### Priority
**Low** - Improves debugging ease

---

## Fix Checklist

| # | File | Issue | Priority | Status |
|---|------|------|----------|------|
| 1 | `custom_xr_egopose_metric.py:214-215` | squeeze() bug | Critical | ✅ Fixed |
| 2 | `mo2cap2_evaluate.py:616-621` | Fragile transpose | Medium | ✅ Fixed |
| 3 | `custom_xr_egopose_metric.py` | Missing shape validation | Low | ✅ Fixed |

> Fix date: 2026-01-21

---

## Testing Methods

### Verifying the squeeze bug
```python
# Run validation with a single sample
val_dataloader = dict(batch_size=1, ...)
# If MPJPE is abnormally large, the bug is present
```

### Post-fix verification
```bash
# Single sample test
python tools/test.py <config> <checkpoint> --cfg-options val_dataloader.batch_size=1
```
