"""Task 24 arm A — replace the lifter's HMD fusion (addition -> cross_attention)
and retrain the WHOLE stack END-TO-END (backbone reopened).

Base = the coarse producer HMD_kinect_v5_flag_cascaded_ground_info_10ep
(CustomEgoposeCascadedRefinementHead_enhanced, ResNet-101 from COCO, 10 ep,
AdamW 5e-4, MultiStepLR[5,8]). Only change: fusion_mode addition ->
cross_attention (z queries hmd, residual + LayerNorm; module already in the
head). No freeze hooks: backbone + encoder + fusion + pose_decoder all train.

Readout is COARSE ego-cam MPJPE (refine bypassed); A must beat arm B
(HMD_kinect_v5_t24_fuse_add, identical but fusion_mode='addition') to credit
the fusion operator rather than the retrain. See HANDOFF.md Task 24 brief.

Launch:
    PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python tools/train.py \
        my_code/custom_config/HMD_kinect_v5_t24_fuse_xattn_config.py \
        --work-dir work_dirs/t24_xattn
"""

_base_ = ['./HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py']

# ---- the one experimental change ----
model = dict(head=dict(fusion_mode='cross_attention'))

# ---- housekeeping: disk + no wandb (matched to arm B) ----
default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=1),
    visualization=dict(enable=False),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

work_dir = 'work_dirs/t24_xattn'
