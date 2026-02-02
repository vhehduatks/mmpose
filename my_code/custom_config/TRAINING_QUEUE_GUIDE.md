# Training Queue Management Guide

## File Location
```
/home/hyeonghwan/github/mmpose/training_queue.txt
```

## Rules

### 1. Append-Only (Add New Entries at Bottom)
- **Always add new experiments at the end of the file**
- Never insert entries in the middle of the queue
- This preserves the chronological order of experiments

```
# BAD - inserting in the middle
[DONE] experiment_1.py
my_code/new_experiment.py    ← Wrong position
[DONE] experiment_2.py
my_code/experiment_3.py

# GOOD - append at bottom
[DONE] experiment_1.py
[DONE] experiment_2.py
my_code/experiment_3.py
my_code/new_experiment.py    ← Correct position
```

### 2. Status Markers
| Marker | Meaning |
|--------|---------|
| (none) | Pending - waiting to be trained |
| `[DONE]` | Completed - training finished |
| `[RUNNING]` | Currently training (optional) |
| `[SKIP]` | Skipped - decided not to run (optional) |

### 3. Comments for Context
Use `#` comments to group related experiments:
```
# Enhanced HMD experiments
my_code/custom_config/HMD_xregopose_enhanced_hmd_torso_ref_full_config.py
my_code/custom_config/HMD_xregopose_enhanced_hmd_ground_ref_full_config.py
```

### 4. Processing Order
- Train experiments from top to bottom
- Mark completed experiments with `[DONE]` prefix
- Do not reorder entries after adding them

## Workflow

### Adding a New Experiment
1. Create the config file
2. (Optional) Run smoke test with small config
3. Add full config path to the **bottom** of `training_queue.txt`
4. Add a comment if it's part of a new experiment group

### Starting Training
1. Find the first entry without `[DONE]` marker
2. Run training: `python tools/train.py <config_path>`
3. Optionally mark as `[RUNNING]` while training

### After Training Completes
1. Add `[DONE]` prefix to the entry
2. Record results in `EXPERIMENT_RESULTS.md`
3. Move to the next pending entry

## Example

```
[DONE] my_code/custom_config/HMD_xregopose_baseline_full_config.py
[DONE] my_code/custom_config/HMD_xregopose_cascaded_refinement_full_config.py
# Enhanced HMD experiments
[DONE] my_code/custom_config/HMD_xregopose_enhanced_hmd_torso_ref_full_config.py
my_code/custom_config/HMD_xregopose_enhanced_hmd_ground_ref_full_config.py
# New architecture experiments
my_code/custom_config/HMD_xregopose_new_architecture_full_config.py
```

## Notes

- The queue serves as a historical record of experiments
- Completed entries should not be deleted (keep for reference)
- If an experiment needs to be re-run, add a new entry at the bottom with a note
