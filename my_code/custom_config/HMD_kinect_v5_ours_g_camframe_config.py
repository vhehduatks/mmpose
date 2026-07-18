"""Task 10 ablation — Ours-G camera-frame direction: skip gravity alignment
(d in the CAMERA frame is a pure function of the pixel — no HMD information
in the direction term; height kept). Control tying to Task 8.9: if
gravity-aligned >> camera-frame, the gravity-anchor story extends to
feature-level conditioning. Everything else identical to the main config.
"""

_base_ = ['./HMD_kinect_v5_ours_g_config.py']

model = dict(head=dict(geo_mode='camera'))
