"""Task 21 combined — joint PE (21.1) + floor-frame sensor tokens (21.2C).
Warm-start from the trained pe checkpoint: joint_pe carries over, the
sensor path is zero-init => init reproduces the pe arm digit-for-digit.
"""

_base_ = ['./HMD_kinect_v5_ours_a_sensC_config.py']

model = dict(head=dict(pe_mode='fixed'))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_pe/best_xregopose_Full Body_All_mpjpe_epoch_2.pth'
