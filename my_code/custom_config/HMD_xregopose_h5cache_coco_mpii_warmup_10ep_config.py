"""
XR EgoPose Training Config with H5 Cached Dataset
- Dual Backbone with COCO + MPII pretrained weights
- Progressive Warmup for Mutual Learning (v2) - 10 Epoch Version

backbone1: COCO pretrained ResNet-101
backbone2: MPII pretrained ResNet-101

Warmup Schedule (10 epochs):
- Epoch 0-1: 0% mutual learning (warmup, backbone 독립 학습)
- Epoch 2-6: 0% → 100% linear ramp (mutual learning 점진적 도입)
- Epoch 7-9: 100% full mutual learning
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
    pretrained_coco = r'F:\download_2\coco_pose_resnet_101_256x192.pth.tar'
    pretrained_mpii = r'F:\download_2\pose_resnet_101_256x256.pth.tar'
    cache_file_train = None
    cache_file_val = None
    cache_file_test = None
else:
    # Linux paths (all on NVMe SSD - no external drive dependency)
    ann_file_train = '/mnt/dataset_vol/h5cache'  # Not used when cache exists
    ann_file_val = '/mnt/dataset_vol/h5cache'    # Not used when cache exists
    ann_file_test = '/mnt/dataset_vol/h5cache'   # Not used when cache exists
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    pretrained_mpii = '/mnt/dataset_vol/pretrained/pose_resnet_101_256x256.pth.tar'
    # Linux cache paths (NVMe SSD - fast I/O!)
    cache_file_train = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'  # Use test cache for val
    cache_file_test = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'

# =============================================================================
# Pretrained Weights
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

# =============================================================================
# Training Config - 10 Epochs
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,  # 10 epoch version (2 warmup + 5 rampup + 3 full)
    val_interval=1,
)
val_cfg = dict()
test_cfg = dict()

# Enable find_unused_parameters for dual backbone model in distributed training
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

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
        milestones=[4, 7],  # LR decay at epoch 4 and 7
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
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
        type='H5CacheVisualizationHook'  # Use H5 cache-aware hook
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
# Custom Hooks - Added MutualLearningWarmupHook
# =============================================================================
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='MutualLearningWarmupHook'),  # Updates head.current_epoch each epoch
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
# Model Architecture - Dual Backbone with COCO + MPII + Progressive Warmup
# =============================================================================
model = dict(
    type='Custom_TopdownPoseEstimator',
    # Backbone 1: COCO pretrained
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=pretrained_coco,
            type='Pretrained'
        ),
        type='ResNet'
    ),
    # Backbone 2: MPII pretrained
    backbone2=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=pretrained_mpii,
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
        type='CustomxRegoposeBaselinel1_multi_backbone_v2',  # v2 with progressive warmup
        decoder=codec,
        in_channels=2048,
        out_channels=16,
        # Progressive warmup settings - 10 epoch version
        mutual_warmup_epochs=2,   # No mutual learning for first 2 epochs
        mutual_rampup_epochs=5,   # Linear increase from 0 to 1 over epochs 2-7
        # Main 2D heatmap loss
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
        loss_backbone_latant=dict(type='MSELoss', loss_weight=1.),  # Scaled by progressive warmup
        loss_backbone_heatmap=dict(
            loss_weight=1.0,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
    ),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False
    ),
)

# =============================================================================
# Data Pipeline (All use H5 cache with embedded images - no disk I/O needed)
# =============================================================================
_meta_keys = ('id', 'img_id', 'img_path', 'category_id', 'crowd_index',
              'ori_shape', 'img_shape', 'input_size', 'input_center',
              'input_scale', 'flip', 'flip_direction', 'flip_indices',
              'raw_ann_info', 'dataset_name', 'action',
              'h5_cache_path', 'h5_img_idx')  # H5 cache keys for visualization

train_pipeline = [
    dict(type='LoadImageFromH5Cache'),  # Load 256x256 image from H5 cache
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),  # Sets input_center/scale
    dict(
        encoder=dict(
            heatmap_size=(47, 47),
            input_size=(256, 256),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'
        ),
        type='GenerateTarget'
    ),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5Cache'),  # Load 256x256 image from H5 cache
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),  # Sets input_center/scale
    dict(
        encoder=dict(
            heatmap_size=(47, 47),
            input_size=(256, 256),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'
        ),
        type='GenerateTarget'
    ),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

test_pipeline = [
    dict(type='LoadImageFromH5Cache'),  # Load 256x256 image from H5 cache
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),  # Sets input_center/scale
    dict(
        encoder=dict(
            heatmap_size=(47, 47),
            input_size=(256, 256),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'
        ),
        type='GenerateTarget'
    ),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
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
    use_cached_images=True,  # Use embedded images from H5 cache
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_val,
    cache_file=cache_file_val,
    rebuild_cache=False,
    use_cached_images=True,  # Use embedded images from H5 cache
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
    use_cached_images=True,  # Use preprocessed images from cache for fast loading
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,  # Uses LoadImageFromH5Cache for fast loading
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
    batch_size=48,  # Reduced from 58 due to dual backbone memory usage
    dataset=dataset_train,
    drop_last=True,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=48,  # Reduced from 58 due to dual backbone memory usage
    dataset=dataset_val,
    drop_last=False,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler')
)

test_dataloader = dict(
    batch_size=64,  # Reduced from 128 due to dual backbone memory usage
    dataset=dataset_test,
    drop_last=False,
    num_workers=0,  # Single process is faster for cached dataset
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
        init_kwargs=dict(project='mmpose_xregopose_coco_mpii_warmup_10ep'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_coco_mpii_warmup_10ep'
