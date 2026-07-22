"""Task 21.2 attribution — HMD token ONLY (controller tokens dropped)."""

_base_ = ['./HMD_kinect_v5_ours_a_sensC_config.py']

model = dict(head=dict(sens_subset='hmd'))
