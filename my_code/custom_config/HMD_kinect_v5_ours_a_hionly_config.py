"""Task 20.1(b) — 47x47 sample ONLY (256 -> 64), no 8x8 context. Isolates
whether the low-res context still matters once the hi-res sample exists.
Fewer params than t01 (capacity-safe direction).
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

model = dict(head=dict(ms_mode='hi_only'))
