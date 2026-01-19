"""
XR EgoPose Training Config with Model Improvements

Improvement #1: PhotometricDistortion for data augmentation
- brightness, contrast, saturation, hue variations
- Note: Spatial augmentations (flip, rotate) disabled for 3D pose compatibility

Improvement #2: Increased Pose Decoder depth
- num_stage: 1 -> 2 (double residual blocks)
- Each Linear block: FC -> BN -> ReLU -> Dropout -> FC -> BN -> ReLU -> Dropout + skip
- Expected: Better 3D pose regression capacity

Expected improvement: 10-20% MPJPE reduction
"""

import platform

# =============================================================================
# Platform Detection & Path Configuration
# =============================================================================
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    ann_file_train = r'F:\ego_cam_dataset\Train'
    ann_file_val = r'F:\ego_cam_dataset\Val'
    ann_file_test = r'F:\ego_cam_dataset\Test'
    pretrained_resnet101 = r'F:\download_2\pose_resnet_101_256x256.pth.tar'
    cache_file_train = None
    cache_file_val = None
    cache_file_test = None
else:
    ann_file_train = '/mnt/sdb2/xr_egopose_full/TrainSet'
    ann_file_val = '/mnt/sdb2/xr_egopose_full/ValSet'
    ann_file_test = '/mnt/sdb2/xr_egopose_full/TestSet'
    pretrained_resnet101 = '/mnt/sdb2/temp/pose_mpii/pose_resnet_101_256x256.pth.tar'
    cache_file_train = '/home/hyeonghwan/h5cache/train_cache.h5'
    cache_file_val = '/home/hyeonghwan/h5cache/val_cache.h5'
    cache_file_test = '/home/hyeonghwan/h5cache/test_cache.h5'

# =============================================================================
# Keypoint flip indices for EgoPose (16 keypoints)
# =============================================================================
# Joint order from egopose_info.py:
# 0:Spine2, 1:Head, 2:LeftArm, 3:LeftForeArm, 4:LeftHand,
# 5:RightArm, 6:RightForeArm, 7:RightHand,
# 8:LeftUpLeg, 9:LeftLeg, 10:LeftFoot, 11:LeftToeBase,
# 12:RightUpLeg, 13:RightLeg, 14:RightFoot, 15:RightToeBase
# flip_indices is defined in egopose_info.py and loaded automatically

# =============================================================================
# Training Config
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,  # Increased for augmented training
    val_interval=1,
)
val_cfg = dict()
test_cfg = dict()

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
    paramwise_cfg=dict(custom_keys={})
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        milestones=[3, 6, 9],
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=3,
        rule='less',
        save_best='xregopose/Full Body_All_mpjpe',
        type='CheckpointHook',
        by_epoch=True
    ),
    visualization=dict(
        enable=True,
        interval=50,
        kpt_thr=0.3,
        type='PoseVisualizationHook'
    )
)

randomness = dict(
    seed=42,
    diff_rank_seed=True,
    deterministic=False
)
resume = False

# =============================================================================
# Codec
# =============================================================================
codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

# =============================================================================
# Custom Hooks
# =============================================================================
custom_hooks = [
    dict(type='SyncBuffersHook'),
]

# =============================================================================
# Environment Config
# =============================================================================
default_scope = 'mmpose'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50
)

# =============================================================================
# Model Architecture (same as before)
# =============================================================================
model = dict(
    type='Custom_TopdownPoseEstimator',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=pretrained_resnet101,
            type='Pretrained'
        ),
        type='ResNet'
    ),
    backbone2=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=pretrained_resnet101,
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
        decoder=codec,
        in_channels=2048,
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=250,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        loss_backbone_latant=dict(type='MSELoss', loss_weight=1.),
        loss_backbone_heatmap=dict(
            loss_weight=1.0,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        out_channels=16,
        type='CustomxRegoposeBaselinel1_multi_backbone',
        # Improvement #2: Increase pose decoder depth (1 -> 2 residual blocks)
        pose_decoder_num_stage=2,
        pose_decoder_linear_size=512,
        pose_decoder_dropout=0.3,
    ),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False
    ),
)

# =============================================================================
# Data Pipeline with Augmentation
# =============================================================================
# Note: Only using PhotometricDistortion for 3D pose estimation.
# RandomFlip and RandomBBoxTransform (rotation) are disabled because they
# would require corresponding transformations of 3D keypoint coordinates.
train_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),

    # Apply affine transformation
    dict(input_size=(256, 256), type='TopdownAffine'),

    # === Data Augmentation (Color only - safe for 3D) ===
    # Photometric distortion: brightness, contrast, saturation, hue
    # This only affects pixel values, not keypoint coordinates
    dict(
        type='PhotometricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=15,
    ),

    # Generate heatmap targets
    dict(
        encoder=dict(
            heatmap_size=(47, 47),
            input_size=(256, 256),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'
        ),
        type='GenerateTarget'
    ),
    dict(type='PackPoseInputs'),
]

# Validation pipeline (no augmentation)
val_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(47, 47),
            input_size=(256, 256),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'
        ),
        type='GenerateTarget'
    ),
    dict(type='PackPoseInputs'),
]

# =============================================================================
# Dataset Config
# =============================================================================
data_mode = 'topdown'
dataset_type = 'H5CachedEgoposeDataset'

dataset_train = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_train,
    cache_file=cache_file_train,
    rebuild_cache=False,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_val,
    cache_file=cache_file_val,
    rebuild_cache=False,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=val_pipeline,
    test_mode=True,
)

dataset_test = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_test,
    cache_file=cache_file_test,
    rebuild_cache=False,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=val_pipeline,
    test_mode=True,
)

# =============================================================================
# DataLoader Config
# =============================================================================
if IS_WINDOWS:
    _num_workers = 2
    _persistent_workers = False
else:
    _num_workers = 6
    _persistent_workers = True

train_dataloader = dict(
    batch_size=58,
    dataset=dataset_train,
    drop_last=True,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=58,
    dataset=dataset_val,
    drop_last=False,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler')
)

test_dataloader = dict(
    batch_size=58,
    dataset=dataset_test,
    drop_last=False,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler')
)

# =============================================================================
# Evaluator
# =============================================================================
val_evaluator = dict(
    ann_file=None,
    type='CustomxRegoposeMetric',
    use_action=False
)

test_evaluator = dict(
    ann_file=None,
    type='CustomxRegoposeMetric',
    use_action=True
)

# =============================================================================
# Visualization
# =============================================================================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(project='mmpose_xregopose_augmented'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_augmented'
