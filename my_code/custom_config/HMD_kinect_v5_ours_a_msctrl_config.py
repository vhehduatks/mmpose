"""Task 20.1 control — EXACT param match to ms_config: the 256-d slot is
the globally-POOLED 47x47 deconv map (same tensor, no per-joint location
information). Isolates per-joint hi-res sampling from added capacity.
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

model = dict(head=dict(ms_mode='fused_ctrl'))
