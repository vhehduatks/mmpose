"""Task 21.2 attribution — controller tokens ONLY (hmd token dropped from
the sensor K/V). Prior: controllers carry the gain (their absolute
floor-frame position is what the relative parametrization discards);
the HMD xyz is doubly redundant (canonicalization + hmd_info).
"""

_base_ = ['./HMD_kinect_v5_ours_a_sensC_config.py']

model = dict(head=dict(sens_subset='ctrl'))
