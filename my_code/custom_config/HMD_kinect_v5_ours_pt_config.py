"""Task 14.2 — Ours-PT (s1+2 +ray): ray-temporal depth context on 14d.

Fine-tunes stage 2 + the temporal branch from the 14d best checkpoint
(backbone + stage 1 frozen via FreezeStage1Hook, so d1 and the heatmap uv
are 14d's — every arm differs only in what the temporal feature carries).
Ours-T fine-tune recipe: 5 ep, AdamW 1e-4, MultiStepLR [3,4]x0.5.

Arms (see ours_pt_modules.py): this config is the main Plücker arm;
_base_ variants override model.head.rays_mode for direction / repcur
(matched-capacity control) / coords (coords-only control).

Init sanity: a val run of this config at init must reproduce 14d's best
val EXACTLY (f_temp enters through zero-init/zero-padded w1 columns).

Launch:
    cd /home/hyeonghwan/github/mmpose && PYTHONPATH=$PWD \
    /home/.conda/envs/mmpose/bin/python tools/train.py \
        my_code/custom_config/HMD_kinect_v5_ours_pt_config.py \
        --work-dir work_dirs/ours_pt
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'
assert not IS_WINDOWS, 'Ours-PT runs on the Linux server only'

custom_imports = dict(
    imports=['my_code.custom_config.ours_pt_modules'],
    allow_failed_imports=False,
)

# =============================================================================
# Data paths
# =============================================================================
data_root_train = '/mnt/dataset_vol/kinect_v5_split/Train'
data_root_test = '/mnt/dataset_vol/kinect_v5_split/Val'
pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
frame_export_root = '/mnt/dataset_vol/frame_export'
kp2d_cache_name = 'kp2d_ours_pilot'

# 14d (Ours-P s1+2) best checkpoint — the base every arm starts from
load_from = ('/home/hyeonghwan/github/mmpose/work_dirs/ours_p_s12/'
             'best_xregopose_Full Body_All_mpjpe_epoch_9.pth')

# =============================================================================
# Training Config (Ours-T fine-tune recipe)
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

_max_epochs = 5

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None

optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW'),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=_max_epochs,
        milestones=[3, 4],
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='xregopose/Full Body_All_mpjpe',
        rule='less',
    ),
    visualization=dict(
        enable=False,
        interval=50,
        kpt_thr=0.3,
        type='PoseVisualizationHook'
    )
)

randomness = dict(seed=42, deterministic=False)
resume = False

# =============================================================================
# Codec (temporal variant: passes the ray-history fields through packing)
# =============================================================================
codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='OursTemporalCodec'
)

# =============================================================================
# Custom Hooks
# =============================================================================
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='FreezeStage1Hook'),
]

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

# =============================================================================
# Environment Config
# =============================================================================
default_scope = 'mmpose'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

log_level = 'INFO'
log_processor = dict(
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50
)

# =============================================================================
# Model
# =============================================================================
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=pretrained_coco,
            type='Pretrained'
        ),
        type='ResNet'
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        type='PoseDataPreprocessor'
    ),
    head=dict(
        type='OursPTHead',
        pinhole_mode='s12',
        rays_mode='plucker',
        in_channels=2048,
        out_channels=16,
        hmd_info_size=12,
        heatmap_decoder_type='efficient',
        decoder=codec,
        # temporal branch (Ours-T shell; f_temp -> stage-2 depth delta)
        f_temp_dim=64,
        history=20,
        temporal_embed=512,
        temporal_heads=32,
        temporal_layers=8,
        zero_f_temp=False,
        # Stage 1 losses
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        # Stage 2 losses
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss'),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss'),
    ),
    test_cfg=dict(
        flip_test=False,
        output_heatmaps=False
    ),
)

# =============================================================================
# Pipeline
# =============================================================================
_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

# =============================================================================
# Dataset Config
# =============================================================================
data_mode = 'topdown'
dataset_type = 'KinectEgoposeTemporalDataset'

dataset_train = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root_train,
    frame_export_root=frame_export_root,
    kp2d_cache_name=kp2d_cache_name,
    history=20,
    ground_info_mode='both_from_ground',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    use_2d_visible=True,
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root_test,
    frame_export_root=frame_export_root,
    kp2d_cache_name=kp2d_cache_name,
    history=20,
    ground_info_mode='both_from_ground',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    use_2d_visible=True,
    pipeline=val_pipeline,
    test_mode=True,
)

# =============================================================================
# DataLoader Config
# =============================================================================
_num_workers = 4
_persistent_workers = True
_batch_size = 16

train_dataloader = dict(
    batch_size=_batch_size,
    dataset=dataset_train,
    drop_last=True,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=_batch_size,
    dataset=dataset_val,
    drop_last=False,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

# =============================================================================
# Evaluator
# =============================================================================
val_evaluator = dict(
    ann_file=None,
    type='CustomxRegoposeMetric',
    use_action=True
)
