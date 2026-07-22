"""Task 21.5 E1-nospat — E1 with the welded appearance removed cleanly:
query = canon_xyz + PE only, sensor cross-attn kept, NO spatial anywhere.
E1 - E1-nospat = the pure worth of welded appearance."""

_base_ = ['./HMD_kinect_v5_ours_a_e1_config.py']

model = dict(head=dict(spatial_in_query=False))
