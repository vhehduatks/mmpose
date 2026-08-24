"""Task 42 / Arm A2 (IEEE VR 2027) — xR-EgoPose (Tome et al.) reproduction, image-only.

BASELINE member of the matched pair with HMD_xregopose_t42_tome_inj_config.py.
Inherits the V2-cache single-stage recipe (10 ep, AdamW 5e-4, MultiStep [4,7],
bs 64, seed 42, ZeroHMDInfo -> 9 zeros) and swaps the head to the Tome-style
`CustomxRegoposeBaselinel1` (heatmap -> autoencoder lifting), i.e. the
architecture of HMD_xregopose_single_coco_full_config.py on the V2 family.
"""

_base_ = ['./HMD_xregopose_single_stage_no_hmd_config.py']

codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

model = dict(
    head=dict(
        _delete_=True,
        type='CustomxRegoposeBaselinel1',
        in_channels=2048,
        out_channels=16,
        decoder=codec,
        loss=dict(loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(loss_weight=250, type='KeypointMSELoss', use_target_weight=False),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
    ),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends)

work_dir = '/mnt/linux_hdd_a/mmpose_work_dirs/t42_xr_tome_base'
