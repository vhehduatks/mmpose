"""Task 19.1(a) — 18b replace block with the stage-2 grid-sample location
taken from a SHARPENED soft-argmax (temperature 0.1 instead of 1.0).
Param-free change; everything else identical to ours_a_replace (10-ep
headline recipe, stage 1 + backbone frozen).
"""

_base_ = ['./HMD_kinect_v5_ours_a_replace_config.py']

model = dict(head=dict(sample_mode='temp', sample_temp=0.1))
