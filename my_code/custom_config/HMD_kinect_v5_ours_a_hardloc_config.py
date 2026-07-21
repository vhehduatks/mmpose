"""Task 19.1(b) — 18b replace block with the stage-2 grid-sample location
taken from hard-argmax + 3x3 value-weighted local mean (removes the
global-tail centroid bias entirely). Param-free; everything else identical
to ours_a_replace (10-ep headline recipe, stage 1 + backbone frozen).
"""

_base_ = ['./HMD_kinect_v5_ours_a_replace_config.py']

model = dict(head=dict(sample_mode='hard_local'))
