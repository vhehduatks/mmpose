"""Task 11 control (MANDATORY, the Ours-G lesson) — identity warp: same
prior channels rendered from the previous frame's 2D but with H=I (no
inertial alignment). Separates "inertial alignment" from "any temporal
prior". Everything else identical to HMD_kinect_v5_ours_2dt_config.py.
"""

_base_ = ['./HMD_kinect_v5_ours_2dt_config.py']

model = dict(head=dict(warp_mode='identity'))
