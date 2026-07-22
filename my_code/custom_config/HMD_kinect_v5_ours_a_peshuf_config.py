"""Task 21.1 control — shuffled-PE: identical nn.Parameter(16,d) but
permuted with a FRESH random order every forward pass. Holds capacity,
destroys only the fixed per-joint identity. The PE claim survives only
if the fixed arm beats this arm.
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

model = dict(head=dict(pe_mode='shuffled'))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_t01/best_xregopose_Full Body_All_mpjpe_epoch_2.pth'
