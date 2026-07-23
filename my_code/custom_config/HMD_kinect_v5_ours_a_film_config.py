"""Task 22a — FiLM-condition spatial_feat on MEASURED ego-cam gravity
(g_ec = R(cam2world)^T @ [0,1,0]; rotation-only, zero translation error).
Channel-wise gamma/beta from zero-init MLPs => init reproduces pesens
(57.70) digit-for-digit. Base = pesens; judge vs the const control.
"""

_base_ = ['./HMD_kinect_v5_ours_a_pesens_config.py']

model = dict(head=dict(film_mode='gravity'))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_pesens/best_xregopose_Full Body_All_mpjpe_epoch_10.pth'
