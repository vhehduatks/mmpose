"""
Enhanced HMD Info - Smoke Test Config

Purpose: Quickly verify that the EnhanceHMDInfo transform and
CustomxRegoposeBaselinel1_enhanced_hmd head work correctly together.

NOTE: keypoint3d is root-relative (head at origin), so only these modes work:
    - 'hands_y': 11 dims (hands Y relative to head)
    - 'torso_reference': 11 dims (torso Y + head-torso distance)
    - 'relative_heights': 12 dims (heights relative to torso)

Usage:
    python tools/train.py my_code/custom_config/HMD_xregopose_enhanced_hmd_small_config.py
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    data_root = r'F:\egodataset_cache\h5cache'
    pretrained_coco = r'F:\egodataset_cache\pose_coco\coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = r'F:\egodataset_cache\h5cache\train_small_1k.h5'
    cache_file_val = r'F:\egodataset_cache\h5cache\val_small_500.h5'
else:
    data_root = '/mnt/dataset_vol/h5cache'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = '/mnt/dataset_vol/h5cache/train_small_1k.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/val_small_500.h5'

# ============================================================
# CHANGE THIS TO TEST DIFFERENT MODES
# ============================================================
_enhance_mode = 'torso_reference'  # Options: 'hands_y', 'torso_reference', 'relative_heights'

# HMD info size mapping
_hmd_size_map = {
    'hands_y': 11,
    'torso_reference': 11,
    'relative_heights': 12,
}
_hmd_info_size = _hmd_size_map[_enhance_mode]
# ============================================================

auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

# Smoke test: 2 epochs, no checkpoint saving
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=2,
        milestones=[1],
        gamma=0.5,
        by_epoch=True
    ),
]

# No checkpoint saving for smoke test
default_hooks = dict(
    checkpoint=None,
    visualization=dict(
        enable=False,
        interval=100,
        kpt_thr=0.3,
        type='H5CacheVisualizationHook'
    )
)

randomness = dict(seed=42, deterministic=False)
resume = False

codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

custom_hooks = [dict(type='SyncBuffersHook')]

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

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

# Model with enhanced HMD head
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
        type='CustomxRegoposeBaselinel1_enhanced_hmd',
        in_channels=2048,
        out_channels=16,
        hmd_info_size=_hmd_info_size,  # Configurable based on mode
        decoder=codec,
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
    ),
    test_cfg=dict(
        flip_test=False,
        output_heatmaps=False
    ),
)

_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'h5_cache_path', 'h5_img_idx'
)

# Pipeline with EnhanceHMDInfo transform
train_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo', mode=_enhance_mode),  # Enhanced HMD info
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo', mode=_enhance_mode),  # Enhanced HMD info
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

data_mode = 'topdown'
dataset_type = 'H5CachedEgoposeDataset'

dataset_train = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root,
    cache_file=cache_file_train,
    rebuild_cache=False,
    use_cached_images=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root,
    cache_file=cache_file_val,
    rebuild_cache=False,
    use_cached_images=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=val_pipeline,
    test_mode=True,
)

if IS_WINDOWS:
    _num_workers = 0
    _persistent_workers = False
    _batch_size = 16
else:
    _num_workers = 4
    _persistent_workers = True
    _batch_size = 32

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

val_evaluator = dict(
    ann_file=None,
    type='CustomxRegoposeMetric',
    use_action=True
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = f'work_dirs/HMD_xregopose_enhanced_hmd_{_enhance_mode}_small'
