"""Task 12.2 control (b) — coords-only: the same temporal shell fed the 2D
coordinate history instead of features. Separates "feature content" from
"any history".
"""

_base_ = ['./HMD_kinect_v5_ours_ft_config.py']

model = dict(head=dict(history_mode='coords'))
