"""Task 18 — self_only WITHOUT floor-frame canonicalization (ablation)."""

_base_ = ['./HMD_kinect_v5_ours_a_config.py']

model = dict(head=dict(attn_mode='self_only', use_canon=False))
