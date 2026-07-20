"""Task 18 — crossonly arm (see ours_a_modules.py docstring)."""

_base_ = ['./HMD_kinect_v5_ours_a_config.py']

model = dict(head=dict(attn_mode='cross_only'))
