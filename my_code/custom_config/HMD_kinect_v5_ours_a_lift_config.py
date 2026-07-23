"""Task 23.1 — gravity-aligned LIFTING: the encoder's heatmap input is
rotated by -roll (g_ec image-plane angle) before pooling; heatmap losses,
codec decode, and stage-2 refine see the raw heatmap. Backbone + heatmap
path frozen (FreezeLiftPathHook); encoder/hmd_linear/pose_decoder/aux +
stage-2 retrain. Base = pesens (57.70 headline), warm-start from its ckpt
(encoder wrap patches the instance forward — checkpoint keys unchanged).
Judge the COARSE against the identity-roll control, not the frozen coarse.
"""

_base_ = ['./HMD_kinect_v5_ours_a_pesens_config.py']

model = dict(head=dict(lift_roll_align=True))

custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='FreezeLiftPathHook'),
]

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_pesens/best_xregopose_Full Body_All_mpjpe_epoch_10.pth'
