"""Task 23.1 CONTROL — identity-roll: same wrap, same grid_sample blur,
angle forced to 0. Isolates "alignment" from "extra lifter training +
interpolation smoothing"."""

_base_ = ['./HMD_kinect_v5_ours_a_lift_config.py']

model = dict(head=dict(lift_roll_zero=True))
