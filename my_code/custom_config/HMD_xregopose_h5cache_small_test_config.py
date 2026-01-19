"""
Small Dataset Config for Quick Training Tests
- Train: 1000 samples
- Val: 500 samples
- For debugging and architecture testing
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    ann_file_train = r'F:\ego_cam_dataset\Train'
    ann_file_val = r'F:\ego_cam_dataset\Val'
    pretrained_coco = r'F:\download_2\coco_pose_resnet_101_256x192.pth.tar'
    cache_file_train = None
    cache_file_val = None
else:
    ann_file_train = '/mnt/dataset_vol/h5cache'
    ann_file_val = '/mnt/dataset_vol/h5cache'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
    # Small datasets for quick testing
    cache_file_train = '/mnt/dataset_vol/h5cache/train_small_1k.h5'
    cache_file_val = '/mnt/dataset_vol/h5cache/val_small_500.h5'

# Training Config - Quick test (2 epochs)
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None  # No test loop for quick training test

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=2, milestones=[1], gamma=0.5, by_epoch=True),
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, type='CheckpointHook'),
    visualization=dict(enable=True, interval=10, kpt_thr=0.3, type='H5CacheVisualizationHook')
)

randomness = dict(seed=42, deterministic=False)
resume = False

# Codec
codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

custom_hooks = [dict(type='SyncBuffersHook')]
default_scope = 'mmpose'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)

# Model - Single backbone COCO for simplicity
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint=pretrained_coco, type='Pretrained'),
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
        loss=dict(loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(loss_weight=250, type='KeypointMSELoss', use_target_weight=False),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        out_channels=16,
        type='CustomxRegoposeBaselinel1'
    ),
    test_cfg=dict(flip_test=False, output_heatmaps=False),
)

# Pipeline
# Extended meta_keys for H5 cache visualization support
_meta_keys = ('id', 'img_id', 'img_path', 'category_id', 'crowd_index',
              'ori_shape', 'img_shape', 'input_size', 'input_center',
              'input_scale', 'flip', 'flip_direction', 'flip_indices',
              'raw_ann_info', 'dataset_name', 'action',
              'h5_cache_path', 'h5_img_idx')

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

# Dataset
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

# DataLoader - Small batch for quick testing
train_dataloader = dict(
    batch_size=32,
    dataset=dataset_train,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=32,
    dataset=dataset_val,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

# Evaluator
val_evaluator = dict(ann_file=None, type='CustomxRegoposeMetric', use_action=False)

# No visualization for quick test
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(name='visualizer', type='CustomPose3dLocalVisualizer_xregopose', vis_backends=vis_backends)

work_dir = 'work_dirs/HMD_xregopose_small_test'
