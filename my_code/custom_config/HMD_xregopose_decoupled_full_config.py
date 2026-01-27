"""
Upper-Lower Decoupled Head - Full Training Config

Key Insight:
    ViT v3: Upper Body 23.49mm (best!), Lower Body 67.19mm (worse)
    Baseline: Upper Body 29.42mm, Lower Body 53.31mm (better)

    → Use ViT v3 for Upper Body (HMD available for head/hands)
    → Use Baseline for Lower Body (no HMD info, Z-vector based)

Architecture:
    Backbone feat [2048, 8, 8]
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  ┌─────────────┐ ┌─────────────┐
  │ Upper Body  │ │ Lower Body  │
  │ (ViT v3)    │ │ (Baseline)  │
  │ 8 joints    │ │ 8 joints    │
  │ +HMD Attn   │ │ Deconv→Z    │
  └─────────────┘ └─────────────┘
        │               │
        └───────┬───────┘
                │
          Concat → [16, 3]

Expected Result:
    Upper: 23.49mm (ViT v3 level)
    Lower: 53.31mm (Baseline level)
    Full: ~38.40mm (vs Baseline 41.37mm, ~3mm improvement!)
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    data_root = r'F:\\egodataset_cache\\h5cache'
    pretrained_coco = r'F:\\egodataset_cache\\pose_coco\\coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = r'F:\\egodataset_cache\\h5cache\\train_cache_with_images.h5'
    cache_file_val = r'F:\\egodataset_cache\\h5cache\\test_cache_with_images.h5'
else:
    data_root = '/mnt/dataset_vol/h5cache'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'

# =============================================================================
# Training Config
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None

# =============================================================================
# Optimizer
# =============================================================================
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# =============================================================================
# LR Schedule - CosineAnnealingLR
# =============================================================================
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    # CosineAnnealingLR
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=10,
    ),
]

# =============================================================================
# Hooks
# =============================================================================
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
        interval=100,
        kpt_thr=0.3,
        type='H5CacheVisualizationHook'
    )
)

randomness = dict(seed=42, deterministic=False)
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
custom_hooks = [dict(type='SyncBuffersHook')]

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

load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50
)

# =============================================================================
# Model - Upper-Lower Decoupled Head
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
        type='CustomEgoposeDecoupledHead',
        in_channels=2048,
        out_channels=16,
        decoder=codec,
        # Upper body (ViT v3 style) params
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        heatmap_size=47,
        # Lower body (Baseline style) params
        z_size=64,
        # Common params
        use_heatmap_recon=True,
        use_refinement=False,
        # Losses
        loss_upper_heatmap_recon=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_lower_heatmap=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_lower_heatmap_recon=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
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

# =============================================================================
# Dataset Config
# =============================================================================
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

# =============================================================================
# DataLoader Config
# =============================================================================
if IS_WINDOWS:
    _num_workers = 0
    _persistent_workers = False
    _batch_size = 32
else:
    _num_workers = 6
    _persistent_workers = True
    _batch_size = 48

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

# =============================================================================
# Visualization - Enable wandb
# =============================================================================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(project='mmpose_xregopose_decoupled'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_decoupled_full'
