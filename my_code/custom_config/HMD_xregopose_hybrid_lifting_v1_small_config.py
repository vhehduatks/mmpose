"""
Hybrid Lifting Head v1 - Smoke Test Config

Baseline Conv Encoder + Attention Refinement
- Conv Encoder: Heatmap → Z [64] (Baseline 동일)
- Z reshape: [64] → [16, 4] (관절당 4-dim)
- Self-Attention: 관절 간 관계 모델링
- Cross-Attention: HMD 정보 선택적 융합

Purpose: Verify training pipeline works (NOT for performance evaluation)
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    data_root = r'F:\egodataset_cache\h5cache'
    pretrained_coco = r'F:\egodataset_cache\pose_coco\coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = r'F:\egodataset_cache\h5cache\train_small_500.h5'
    cache_file_val = r'F:\egodataset_cache\h5cache\val_small_100.h5'
else:
    data_root = '/mnt/dataset_vol/h5cache'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = '/mnt/dataset_vol/h5cache/train_small_1k.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/val_small_500.h5'

# =============================================================================
# Training Config - Smoke Test (1 epoch)
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        milestones=[4, 7],
        gamma=0.5,
        by_epoch=True,
    ),
]

# =============================================================================
# Hooks - No checkpoint saving for smoke test
# =============================================================================
default_hooks = dict(
    checkpoint=dict(
        save_best=None,
        type='CheckpointHook',
        interval=999,
    ),
    visualization=dict(
        enable=False,
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

# =============================================================================
# Model - Hybrid Lifting v1 (Conv Encoder + Attention Refinement)
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
        type='CustomEgoposeHybridLiftingHead',
        in_channels=2048,
        out_channels=16,
        decoder=codec,
        # Hybrid Lifting params
        joint_dim=64,
        num_heads=4,
        num_self_attn_layers=2,
        dropout=0.1,
        # Losses (Baseline 동일)
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_heatmap_recon=dict(
            loss_weight=250,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_hmd=dict(type='MSELoss', loss_weight=1.0),
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
    'h5_cache_path', 'h5_img_idx'
)

train_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
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
else:
    _num_workers = 4
    _persistent_workers = True

train_dataloader = dict(
    batch_size=32,
    dataset=dataset_train,
    drop_last=True,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=32,
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
    use_action=False
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_hybrid_lifting_v1_small'
