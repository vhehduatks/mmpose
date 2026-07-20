"""Task 18 — mlpmatch arm (see ours_a_modules.py docstring)."""

_base_ = ['./HMD_kinect_v5_ours_a_config.py']

model = dict(head=dict(attn_mode='mlp_matched'))
