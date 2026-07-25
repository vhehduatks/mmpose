"""Task 26.2a — the CORRECT comparison: pesens's floor stage-2 retrained
END-TO-END, matched to Task 26 arm A's regime. A's true opponent.

26.1a compared A's ego-cam stage-2 vs floor on a FROZEN backbone — but a frozen
backbone is never neutral (it disarms A's co-adapted-backbone advantage while
leaving floor on features it suits). The missing cell is floor's DESIGN under
A's exact end-to-end regime. This config = the deployed pesens floor stage-2
(OursAttnCascadedHead: attn_mode='replace', sensor_mode='floor', pe_mode='fixed',
global K/V [z,pose,kin,hmd], position-only floor sensors) with:
  - backbone REOPENED (FreezeStage1Hook dropped; stage 1 trainable)
  - headline-lifter init + FRESH stage-2 (out zero-init) — exactly as Task 26 A
  - same 10 ep / AdamW 5e-4 / MultiStepLR [5,8] (already the pesens recipe)
so the ONLY variable vs Task 26 A (58.79) is the stage-2 design.

Readout: refined vs Task 26 A **58.79** (design at matched regime) and vs pesens
**57.70** (regime tax, same design frozen-vs-reopened). If floor-e2e ~58.x, A's
ego-cam design is exonerated (both pay the ~1 mm reopen tax; headline wins only
by staying frozen). If ~57.x, floor tolerates end-to-end better.

Launch:
    PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python tools/train.py \
        my_code/custom_config/HMD_kinect_v5_t262a_config.py --work-dir work_dirs/t262a
"""

_base_ = ['./HMD_kinect_v5_ours_a_pesens_config.py']

# headline-lifter init + fresh stage-2 (matches Task 26 A); recipe already
# 10ep/5e-4/[5,8] in the pesens base.
load_from = ('/mnt/dataset_vol/work_dir_260408/'
             'HMD_kinect_v5_flag_cascaded_ground_info_10ep/'
             'best_xregopose_Full Body_All_mpjpe_epoch_10.pth')

# REOPEN the backbone: drop FreezeStage1Hook, keep SyncBuffers only
custom_hooks = [dict(type='SyncBuffersHook')]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='xregopose/Full Body_All_mpjpe', rule='less'),
)

work_dir = 'work_dirs/t262a'
