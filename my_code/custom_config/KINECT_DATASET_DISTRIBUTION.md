# Kinect V5 Flag Dataset — Per-Action Frame Distribution

> Dataset: `egodataset_flag_fixed_ver5` with `use_2d_visible=True`
> Train: `/mnt/dataset_vol/kinect_v5_split/Train` (32 batches)
> Val: `/mnt/dataset_vol/kinect_v5_split/Val` (9 batches)
> Split: batch-level 8:2

## Summary

| Split | Sessions | Frames |
|-------|----------|--------|
| Train | 642 | 105,678 |
| Val | 182 | 26,401 |
| **Total** | **824** | **132,079** |

## Per-Action Distribution

| Action | Train Sessions | Train Frames | Val Sessions | Val Frames | Total Frames | Train % | Val % |
|--------|---------------|-------------|-------------|-----------|-------------|---------|-------|
| Dancing1 | 32 | 5,301 | 9 | 1,330 | 6,631 | 5.02% | 5.04% |
| Dancing2 | 32 | 5,302 | 9 | 1,328 | 6,630 | 5.02% | 5.03% |
| Dancing3 | 32 | 5,288 | 9 | 1,322 | 6,610 | 5.00% | 5.01% |
| Gaming-Archery | 32 | 5,129 | 9 | 1,311 | 6,440 | 4.85% | 4.97% |
| Gaming-Baseball | 35 | 5,584 | 10 | 1,287 | 6,871 | 5.28% | 4.87% |
| Gaming-Boxing | 32 | 5,539 | 9 | 1,312 | 6,851 | 5.24% | 4.97% |
| Gaming-Golf | 33 | 5,448 | 9 | 1,309 | 6,757 | 5.16% | 4.96% |
| Gaming-Shooting | 33 | 5,595 | 9 | 1,305 | 6,900 | 5.29% | 4.94% |
| Greeting-ShakingHand | 32 | 5,269 | 9 | 1,312 | 6,581 | 4.99% | 4.97% |
| Greeting-WavingHand | 32 | 5,240 | 9 | 1,306 | 6,546 | 4.96% | 4.95% |
| Patting | 32 | 5,254 | 9 | 1,314 | 6,568 | 4.97% | 4.98% |
| Reacting-Cheering | 34 | 5,594 | 9 | 1,316 | 6,910 | 5.29% | 4.98% |
| Reacting-Clapping | 30 | 4,953 | 9 | 1,312 | 6,265 | 4.69% | 4.97% |
| Reacting-Yelling | 31 | 5,115 | 9 | 1,325 | 6,440 | 4.84% | 5.02% |
| Talking | 32 | 5,249 | 9 | 1,315 | 6,564 | 4.97% | 4.98% |
| UpperStreching | 32 | 5,236 | 9 | 1,314 | 6,550 | 4.95% | 4.98% |
| Walking | 33 | 5,440 | 9 | 1,325 | 6,765 | 5.15% | 5.02% |
| Workout-BicelCurl | 31 | 5,060 | 9 | 1,299 | 6,359 | 4.79% | 4.92% |
| Workout-FrontRaise | 32 | 5,250 | 9 | 1,307 | 6,557 | 4.97% | 4.95% |
| Workout-KettleBell | 30 | 4,832 | 10 | 1,452 | 6,284 | 4.57% | 5.50% |
| **Total** | **642** | **105,678** | **182** | **26,401** | **132,079** | | |

## Observations

- The dataset is **well-balanced**: each action accounts for 4.57–5.29% of training frames (ideal = 5.00%).
- Smallest action: Workout-KettleBell (4,832 train frames, 30 sessions)
- Largest action: Gaming-Shooting (5,595 train frames, 33 sessions)
- Val set is even more uniform (4.87–5.50% per action) due to the fixed 9 batches per action.
- Average ~165 frames per session (train), ~145 frames per session (val).
