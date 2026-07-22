"""Task 21.1 — learnable joint positional embedding (zero-init, added to
the joint tokens before self-attn; flows into the cross-attn query). Base
= t01 (18b replace + tau=0.1 sampling); warm-start FROM the t01
checkpoint — zero-init PE reproduces t01 digit-for-digit at init.
Judge vs the shuffled-PE control, not t01.
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

model = dict(head=dict(pe_mode='fixed'))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_t01/best_xregopose_Full Body_All_mpjpe_epoch_2.pth'
