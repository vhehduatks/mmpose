"""Task 20.1(a) — multi-scale spatial feature: 8x8 backbone sample (2048)
+ per-joint 47x47 add_deconv sample (256), concatenated -> 64-d proj.
Base = t01 (18b replace block + tau=0.1 sampling); only the feature
source changes. Judge vs the param-matched pooled control (msctrl).
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

model = dict(head=dict(ms_mode='fused'))
