"""
XR EgoPose Training Config with H5 Cached Dataset

This config uses H5CachedEgoposeDataset for fast dataset loading.
Instead of parsing 65k+ JSON files at init, it loads from a single H5 cache file.

Before training, build the cache:
    # Windows
    python tools/dataset_converters/build_egopose_h5cache.py \
        --data-root F:/ego_cam_dataset/Train --num-workers 4

    # Linux
    python tools/dataset_converters/build_egopose_h5cache.py \
        --data-root /mnt/sdb2/xr_egopose_full/TrainSet --num-workers 4

Expected loading time improvement:
    - Original: ~5-10 minutes
    - With H5 cache: ~3-5 seconds
"""

import platform

# =============================================================================
# Platform Detection & Path Configuration
# =============================================================================
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    # Windows paths
    ann_file_train = r'F:\ego_cam_dataset\Train'
    ann_file_val = r'F:\ego_cam_dataset\Val'
    ann_file_test = r'F:\ego_cam_dataset\Test'
    pretrained_resnet101 = r'F:\download_2\pose_resnet_101_256x256.pth.tar'
else:
    # Linux paths
    ann_file_train = '/mnt/sdb2/xr_egopose_full/TrainSet'
    ann_file_val = '/mnt/sdb2/xr_egopose_full/ValSet'
    ann_file_test = '/mnt/sdb2/xr_egopose_full/TestSet'
    pretrained_resnet101 = '/mnt/sdb2/temp/pose_mpii/pose_resnet_101_256x256.pth.tar'

# Cache file paths (auto-generated if not specified)
# These will be created at: {ann_file_xxx}/annotations_cache.h5
cache_file_train = None  # Will auto-create at ann_file_train/annotations_cache.h5
cache_file_val = None
cache_file_test = None

# =============================================================================
# Pretrained Weights
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

# =============================================================================
# Training Config
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1,
)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
    paramwise_cfg=dict(
        custom_keys={}
    )
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        milestones=[i for i in range(1, 10)],
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
        interval=3,
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
# Codec (Heatmap Encoding)
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
# Model Architecture
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
        # Main 2D heatmap loss
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        # λ_cos = 0.1 (Section IV-A)
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        # λ_recon = 250 (Section IV-A) - Reconstruction loss
        loss_heatmap_recon=dict(
            loss_weight=250,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        # λ_L1 = 0.25 (Section IV-A) - L1 norm loss
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        # L2 norm loss
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        # HMD reconstruction loss
        loss_hmd=dict(type='MSELoss'),
        # Backbone latent feature alignment loss
        loss_backbone_latant=dict(type='MSELoss', loss_weight=1.),
        # λ_sub = 1.0 (Section IV-A) - Sub-backbone heatmap loss
        loss_backbone_heatmap=dict(
            loss_weight=1.0,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        out_channels=16,
        type='CustomxRegoposeBaselinel1_multi_backbone'
    ),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False
    ),
)

# =============================================================================
# Data Pipeline
# =============================================================================
train_pipeline = [
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
# Dataset Config - Using H5CachedEgoposeDataset for FAST loading
# =============================================================================
data_mode = 'topdown'
dataset_type = 'H5CachedEgoposeDataset'  # <-- Key change: use cached dataset

dataset_train = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_train,
    cache_file=cache_file_train,  # Auto-generated if None
    rebuild_cache=False,          # Set True to force rebuild
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
train_dataloader = dict(
    batch_size=58,
    dataset=dataset_train,
    drop_last=True,
    num_workers=6,
    persistent_workers=True,  # Can enable with fast loading
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=58,
    dataset=dataset_val,
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler')
)

test_dataloader = dict(
    batch_size=58,
    dataset=dataset_test,
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
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
        init_kwargs=dict(project='mmpose_xregopose_h5cache'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_h5cache'
