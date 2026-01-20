"""
XR EgoPose Training Config - Single Backbone + Soft-argmax Lifting

Architecture:
    ResNet-101 (COCO pretrained) → Heatmap [16, 47, 47]
                                        ↓
    soft_argmax → 2D coords [16, 2] + confidence [16]
                                        ↓
    Lifting Network (2D + conf + HMD → 3D)
                                        ↓
    3D Pose [16, 3]

Key Features:
- Single backbone (vs dual backbone in other configs)
- Heatmap trained with Gaussian MSE (dense supervision)
- Soft-argmax for sub-pixel 2D coordinate extraction
- Lightweight Lifting network (~4M params vs 40M HeatmapDecoder)
- No HeatmapEncoder/Decoder bottleneck

Losses:
- loss_heatmap: MSE(pred_heatmap, gt_heatmap) - weight 1000
- loss_coord: MSE(soft_argmax coords, gt_coords) - weight 10
- loss_pose_l2norm: L2(pred_3d, gt_3d) - weight 1.0
- loss_cosine_similarity - weight 0.1
- loss_limb_length - weight 0.25

Comparison with other configs:
- HMD_xregopose_single_coco_full_config.py: Uses HeatmapEncoder→Z→HeatmapDecoder (40M params)
- This config: Uses soft_argmax→Lifting (4M params)
"""

import platform

# =============================================================================
# Platform Detection & Path Configuration
# =============================================================================
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    ann_file_train = r'F:\\ego_cam_dataset\\Train'
    ann_file_val = r'F:\\ego_cam_dataset\\Val'
    ann_file_test = r'F:\\ego_cam_dataset\\Test'
    pretrained_coco = r'F:\\download_2\\coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = None
    cache_file_val = None
    cache_file_test = None
else:
    ann_file_train = '/mnt/dataset_vol/h5cache'
    ann_file_val = '/mnt/dataset_vol/h5cache'
    ann_file_test = '/mnt/dataset_vol/h5cache'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = '/mnt/dataset_vol/h5cache/train_cache_with_images.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'
    cache_file_test = '/mnt/dataset_vol/h5cache/test_cache_with_images.h5'

# =============================================================================
# Training Config
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,
    val_interval=1,
)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        milestones=[8, 12],
        gamma=0.1,
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
        type='H5CacheVisualizationHook'
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
# Model Architecture - Single Backbone with Lifting Head
# =============================================================================
model = dict(
    type='TopdownPoseEstimator',  # Standard single backbone
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
        type='CustomEgoposeLiftingHead',
        decoder=codec,
        in_channels=2048,
        out_channels=16,
        # Heatmap generation layers
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        deconv_stride_sizes=(2, 2),
        # Losses
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_coord=dict(
            type='MSELoss',
            loss_weight=10.0
        ),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        # Lifting network config
        lifting_hidden_dim=1024,
        lifting_num_blocks=2,
        lifting_dropout=0.3,
        # Soft-argmax config
        soft_argmax_temperature=1.0,
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
_meta_keys = ('id', 'img_id', 'img_path', 'category_id', 'crowd_index',
              'ori_shape', 'img_shape', 'input_size', 'input_center',
              'input_scale', 'flip', 'flip_direction', 'flip_indices',
              'raw_ann_info', 'dataset_name', 'action',
              'h5_cache_path', 'h5_img_idx')

train_pipeline = [
    dict(type='LoadImageFromH5Cache'),
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
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5Cache'),
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
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

test_pipeline = [
    dict(type='LoadImageFromH5Cache'),
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
    use_cached_images=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=ann_file_val,
    cache_file=cache_file_val,
    rebuild_cache=False,
    use_cached_images=True,
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
    use_cached_images=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
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
    batch_size=64,  # Larger batch possible (smaller model)
    dataset=dataset_train,
    drop_last=True,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=64,
    dataset=dataset_val,
    drop_last=False,
    num_workers=_num_workers,
    persistent_workers=_persistent_workers,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler')
)

test_dataloader = dict(
    batch_size=128,
    dataset=dataset_test,
    drop_last=False,
    num_workers=0,
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
        init_kwargs=dict(project='mmpose_xregopose_single_lifting'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_single_lifting'
