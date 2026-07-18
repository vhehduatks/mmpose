"""Task 12.2 control (a) — replicate-current: same architecture, history
replaced by the current frame's features repeated H times. Separates
"temporal information" from "extra parameters/capacity".
"""

_base_ = ['./HMD_kinect_v5_ours_ft_config.py']

model = dict(head=dict(history_mode='replicate'))
