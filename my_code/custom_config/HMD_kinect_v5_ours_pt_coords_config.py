"""Task 14.2 — Ours-PT coords arm (see ours_pt_modules.py docstring)."""

_base_ = ['./HMD_kinect_v5_ours_pt_config.py']

model = dict(head=dict(rays_mode='coords'))
