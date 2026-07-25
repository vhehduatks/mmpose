"""Task 26.1a — arm A's ego-cam stage-2 mounted on the DEPLOYED frozen base.

The single-variable comparison vs pesens. Task 26 arm A was a 3-variable change
(stage-2 design + backbone reopen + fusion removal) and lost only because it
started 1.0 mm behind at coarse (68.19 vs 67.19). Here: FREEZE backbone +
stage 1 (coarse == deployed 67.19), RESTORE the addition fusion (free on a
frozen stage 1), keep arm A's ego-cam stage-2 (dual-frame tokens + invariant
bias + dual-frame 6DoF controller tokens). Only the stage-2 operator differs
from pesens now.

Recipe matched to pesens: 5 ep, AdamW 1e-4, MultiStepLR [3,4]x0.5. Warm-start
the frozen stage-1 from the headline lifter (coarse 67.19); stage-2 is new,
out zero-init (no digit-identity possible — architecture differs, same as
`replace`).

GATE 26.1a (0 GPU, before training): out zero-init + frozen stage-1 =>
refined == coarse == 67.19 exactly.

Readout: refined vs pesens 58.08 internal / 57.70 common on the SAME coarse.
This single number decides whether Task 26's mechanism is deployable.

Launch:
    PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python tools/train.py \
        my_code/custom_config/HMD_kinect_v5_t261a_config.py --work-dir work_dirs/t261a
"""

_base_ = ['./HMD_kinect_v5_t26_ecA_config.py']

# frozen deployed base: coarse == 67.19 (headline lifter), fusion RESTORED
load_from = ('/mnt/dataset_vol/work_dir_260408/'
             'HMD_kinect_v5_flag_cascaded_ground_info_10ep/'
             'best_xregopose_Full Body_All_mpjpe_epoch_10.pth')

model = dict(head=dict(restore_fusion=True))   # A's stage-2, addition fusion back

custom_hooks = [dict(type='SyncBuffersHook'), dict(type='FreezeStage1Hook')]

# pesens fine-tune recipe (frozen stage 1): 5 ep, 1e-4, [3,4]x0.5
_max_epochs = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epochs,
                 val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.0001, type='AdamW'))
param_scheduler = [dict(type='MultiStepLR', begin=0, end=_max_epochs,
                        milestones=[3, 4], gamma=0.5, by_epoch=True)]

work_dir = 'work_dirs/t261a'
