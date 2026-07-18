"""Task 10 ablation — Ours-G direction-only: drop the height term
(per-joint geo feature = d_f (3) only) to isolate what h adds.
Everything else identical to HMD_kinect_v5_ours_g_config.py.
"""

_base_ = ['./HMD_kinect_v5_ours_g_config.py']

model = dict(head=dict(use_height=False))
