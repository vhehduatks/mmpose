"""Task 21.4 — fusion-off + controller tokens only (checks controllers
still carry the rest under the clean condition; expect ~= B-all 58.95)."""

_base_ = ['./HMD_kinect_v5_ours_a_sensB_config.py']

model = dict(head=dict(sens_subset='ctrl'))
