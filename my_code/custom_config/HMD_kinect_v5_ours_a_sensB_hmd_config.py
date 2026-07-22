"""Task 21.4 — THE DECISIVE ARM: fusion-off + hmd token only. GBH is
zeroed (use_hmd=False), so the hmd token's height is the ONLY head height
the model gets. Gain over B-none => the degenerate-token/redundancy
explanation of hmd inertness is confirmed."""

_base_ = ['./HMD_kinect_v5_ours_a_sensB_config.py']

model = dict(head=dict(sens_subset='hmd'))
