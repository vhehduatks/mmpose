"""Task 14 — Ours-P (s12) variant (see ours_p_modules.py docstring)."""

_base_ = ['./HMD_kinect_v5_ours_p_s2_config.py']

model = dict(head=dict(pinhole_mode='s12'))
