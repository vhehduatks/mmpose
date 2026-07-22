"""Task 21.4 — fusion-off NO-SENSOR base (vision + canonicalization only).
Arm B's exact config (use_hmd=False, FreezeBackboneHook, full stage-1+2
retrain, PE off) with sensor_mode='off'. Canonicalization stays ON (reads
m2w from labels, independent of use_hmd)."""

_base_ = ['./HMD_kinect_v5_ours_a_sensB_config.py']

model = dict(head=dict(sensor_mode='off'))
