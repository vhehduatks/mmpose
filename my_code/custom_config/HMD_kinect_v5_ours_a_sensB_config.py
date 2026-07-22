"""Task 21.2 arm B — the PLACEMENT-MOVE hypothesis: stage-1 hmd fusion OFF
(use_hmd=False zeroes ALL 12 hmd_info dims incl. GBH — flag semantics
checked and reported), device sensors consumed ONLY as floor-frame stage-2
cross-attn tokens. Stage 1 must retrain (its input distribution changes):
FreezeBackboneHook replaces FreezeStage1Hook — backbone frozen, everything
else trains with the headline 10-ep recipe from the headline ckpt
(deviation disclosed). B > A(t01) => sensors belong in the geometry
domain; B falls toward 71 => stage-1 fusion irreplaceable.
"""

_base_ = ['./HMD_kinect_v5_ours_a_sensC_config.py']

model = dict(head=dict(sensor_mode='floor', use_hmd_token=False))

train_dataloader = dict(dataset=dict(use_hmd=False))
val_dataloader = dict(dataset=dict(use_hmd=False))

custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='FreezeBackboneHook'),
]

load_from = '/mnt/dataset_vol/work_dir_260408/HMD_kinect_v5_flag_cascaded_ground_info_10ep/best_xregopose_Full Body_All_mpjpe_epoch_10.pth'
