# xR-EgoPose Per-Action Results (Official 9-Category Grouping)

> Checkpoint: `best_xregopose_Full Body_All_mpjpe_epoch_8.pth`
> Test set: 115,621 samples, 43 actions grouped into 9 categories
> Evaluation protocol: Official xR-EgoPose grouping

## Grouped Results (MPJPE in mm)

| Category | Full Body | Upper Body | Lower Body | Samples |
|----------|-----------|------------|------------|---------|
| Gesticulating | 34.97 | 34.55 | 35.51 | 5,205 |
| Reacting | 33.18 | 29.84 | 37.48 | 6,790 |
| Greeting | 32.57 | 32.93 | 32.11 | 1,855 |
| Talking | 29.89 | 28.29 | 31.94 | 3,085 |
| UpperStretching | 34.48 | 28.00 | 42.82 | 61,163 |
| Gaming | 34.21 | 30.53 | 38.95 | 3,985 |
| LowerStretching | 32.38 | 23.99 | 43.17 | 18,574 |
| Patting | 43.32 | 36.75 | 51.76 | 3,002 |
| Walking | 33.56 | 26.70 | 42.39 | 11,962 |
| **All** | **34.06** | **28.03** | **41.82** | **115,621** |

## Individual Action Results (sorted by MPJPE)

| Action | Group | MPJPE (mm) | Samples |
|--------|-------|------------|---------|
| Surprised | Reacting | 25.49 | 540 |
| Beckoning | Reacting | 26.01 | 330 |
| Talking | Talking | 27.95 | 1,485 |
| Clapping | Reacting | 27.96 | 135 |
| Revealing_Dice | Gaming | 28.09 | 555 |
| Angry_Point | Reacting | 28.36 | 315 |
| Reaching_Out | Talking | 28.59 | 570 |
| Shaking_Hands_2 | Greeting | 28.66 | 375 |
| Tpose_Take_001 | UpperStretching | 28.90 | 5 |
| Pointing | Gesticulating | 28.98 | 520 |
| Crazy_Gesture | Gesticulating | 29.42 | 725 |
| Petting | Patting | 29.67 | 410 |
| Arm_Gesture | Gesticulating | 29.75 | 440 |
| Hands_Forward_Gesture | Greeting | 29.96 | 470 |
| Dismissing_Gesture | Reacting | 30.24 | 460 |
| Fist_Pump | Gesticulating | 30.24 | 525 |
| Happy_Hand_Gesture | Gesticulating | 30.29 | 380 |
| Thinking | UpperStretching | 30.80 | 620 |
| Loser | Reacting | 32.09 | 470 |
| lower_stretching | LowerStretching | 32.36 | 18,014 |
| Sitting_Disapproval | LowerStretching | 32.77 | 560 |
| Insult | Talking | 32.80 | 330 |
| Hand_Raising | Gesticulating | 33.09 | 555 |
| Golf_Putt_Failure | Gaming | 33.22 | 1,425 |
| Terrified | Reacting | 33.43 | 2,925 |
| walking | Walking | 33.46 | 11,700 |
| No | Talking | 33.68 | 700 |
| Strong_Gesture | Reacting | 34.41 | 240 |
| upper_stretching | UpperStretching | 34.50 | 60,243 |
| Standing_Greeting | Greeting | 34.93 | 645 |
| Charge | Gaming | 35.04 | 800 |
| Quick_Formal_Bow | Greeting | 35.79 | 365 |
| Taunt_Gesture | Gesticulating | 36.47 | 270 |
| Weight_Shift_Gesture | Walking | 37.22 | 250 |
| Golf_Putt_Victory__1_ | Gaming | 37.66 | 1,205 |
| Standing_1H_Magic_Attack_01 | UpperStretching | 39.56 | 295 |
| Rallying | Reacting | 39.62 | 885 |
| Pointing_Gesture | Reacting | 40.03 | 270 |
| Pain_Gesture | Reacting | 42.44 | 220 |
| Counting__1_ | Gesticulating | 42.97 | 1,790 |
| Petting_Animal | Patting | 45.48 | 2,592 |
| anim_Clip1 | Walking | 61.70 | 12 |

## Paper Table Row

| Method | Gaming | Gestic. | Greeting | LowerStr. | Patting | Reacting | Talking | UpperStr. | Walking | All |
|--------|--------|---------|----------|-----------|---------|----------|---------|-----------|---------|-----|
| **Ours** | 34.2 | 35.0 | 32.6 | 32.4 | 43.3 | 33.2 | 29.9 | 34.5 | 33.6 | **34.1** |
