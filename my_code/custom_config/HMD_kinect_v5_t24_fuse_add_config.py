"""Task 24 arm B (CONTROL) — same END-TO-END retrain as arm A but fusion_mode
stays 'addition' (the current design). Isolates the retrain-alone gain: arm A
(cross_attention) must beat THIS, not the old frozen coarse, so A-B is the pure
fusion-operator effect. This is also the honest end-to-end number for the
current addition design under matched conditions. See HANDOFF.md Task 24 brief.

Launch:
    PYTHONPATH=$PWD /home/.conda/envs/mmpose/bin/python tools/train.py \
        my_code/custom_config/HMD_kinect_v5_t24_fuse_add_config.py \
        --work-dir work_dirs/t24_add
"""

_base_ = ['./HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py']

# addition is the base default; set explicitly for symmetry with arm A.
model = dict(head=dict(fusion_mode='addition'))

default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=1),
    visualization=dict(enable=False),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

work_dir = 'work_dirs/t24_add'
