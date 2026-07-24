"""Task 26 arm C (BIAS CONTROL — the decisive mechanism control) — identical
ego-cam architecture to arm A with the rotation-invariant relational bias
switched OFF. C is the end-to-end re-measurement of nocanon/E2; A-C is the
pure value of the invariant bias (a gain cannot otherwise be separated from
"end-to-end retraining helped"). See ours_ec_modules.py, HANDOFF.md Task 26.

    stage2_frame='egocam', use_bias=False, use_sensor_ego=True, use_sensor_rot=True
"""

_base_ = ['./HMD_kinect_v5_t26_ecA_config.py']

model = dict(head=dict(use_bias=False))
work_dir = 'work_dirs/t26_ecC'
