"""Task 26 arm B (FRAME CONTROL) — coordinate-transform alignment (floor/canon
frame, Task-18 style) instead of the invariant bias, floor-frame sensors,
same end-to-end budget. A vs B answers "does invariant-bias alignment in the
lifting frame match coordinate-transform alignment?" (what frozen-backbone
nocanon/E2 could not). See ours_ec_modules.py, HANDOFF.md Task 26.

    stage2_frame='floor', use_bias=False, use_sensor_ego=False, use_sensor_rot=True
"""

_base_ = ['./HMD_kinect_v5_t26_ecA_config.py']

model = dict(head=dict(stage2_frame='floor', use_bias=False,
                       use_sensor_ego=False, use_sensor_rot=True))
work_dir = 'work_dirs/t26_ecB'
