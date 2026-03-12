"""
Mo2Cap2 Cascaded Refinement + BOTH FROM GROUND - SMOKE TEST

Smoke test config to verify:
1. 2D keypoint visualization fix (bbox 1024x1024 for test images)
2. BGR→RGB conversion for WandB
3. Coordinate transformation in visualizer

Settings:
- 1 epoch training
- sample_interval=100 (reduces ~530k to ~5.3k samples)
- Frequent visualization (every 5 val iters, every 50 train iters)
- No checkpoint saving
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    data_root = r'F:\mo2cap2_dataset\training_data'
    test_data_root = r'F:\mo2cap2_dataset\test_data\TestSet'
    pretrained_coco = r'F:\egodataset_cache\pose_coco\coco_pose_resnet_101_256x192.pth.tar'
else:
    data_root = '/mnt/sdb2/mo2cap2_dataset/training_data'
    test_data_root = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'

auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

# Smoke test: 1 epoch only
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
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
        end=1,
        milestones=[],
        gamma=0.1,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=None,  # No checkpoint for smoke test
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,
        interval=5,  # Val: visualize frequently to verify fix
        train_interval=50,  # Train: every 50 iters
    ),
    logger=dict(type='LoggerHook', interval=10),
)

randomness = dict(seed=42, deterministic=False)
resume = False

MO2CAP2_SKELETON = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13), (13, 14),
]

MO2CAP2_SYMMETRIC_LIMBS = [
    ((0, 4), (0, 1)), ((4, 5), (1, 2)), ((5, 6), (2, 3)),
    ((0, 11), (0, 7)), ((11, 12), (7, 8)), ((12, 13), (8, 9)), ((13, 14), (9, 10)),
]

codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

custom_hooks = []
default_scope = 'mmpose'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)

# Model: hmd_info_size=12 (9 base + 3 ground ref)
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
        type='CustomMo2Cap2CascadedRefinementHead_enhanced',
        in_channels=2048,
        out_channels=15,
        hmd_info_size=12,
        heatmap_decoder_type='efficient',
        decoder=codec,
        loss=dict(loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity', skeleton=MO2CAP2_SKELETON),
        loss_heatmap_recon=dict(loss_weight=500, type='KeypointMSELoss', use_target_weight=False),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss', skeleton=MO2CAP2_SKELETON),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss', symmetric_limbs=MO2CAP2_SYMMETRIC_LIMBS),
    ),
    test_cfg=dict(flip_test=False, output_heatmaps=False),
)

_train_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'h5_chunk_idx', 'h5_local_idx', 'h5_chunk_path'
)

_val_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'frame_idx', 'sequence_name'
)

# Pipeline with both_from_ground mode
train_pipeline = [
    dict(type='LoadImageFromH5'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_train_meta_keys),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='Mo2Cap2CenterCrop', margin_left=128, margin_right=128),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_val_meta_keys),
]

data_mode = 'topdown'

# Smoke test: sample_interval=100 reduces dataset significantly
dataset_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=train_pipeline,
    input_size=(256, 256),
    sample_interval=100,  # ~530k / 100 = ~5.3k samples
    use_zoom=False,
)

dataset_val = dict(
    type='Mo2Cap2CombinedTestDataset',
    data_root=test_data_root,
    data_mode=data_mode,
    pipeline=val_pipeline,
    input_size=(256, 256),
    test_mode=True,
)

if IS_WINDOWS:
    _num_workers = 0
    _persistent_workers = False
    _batch_size = 16
else:
    _num_workers = 4
    _persistent_workers = True
    _batch_size = 32  # Smaller batch for smoke test

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

val_evaluator = dict(ann_file=None, type='CustomMo2Cap2Metric', use_action=True)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mo2cap2-pose-estimation',
            name='smoke_test_2d_keypoint_fix',
            tags=['mo2cap2', 'smoke_test', '2d_keypoint_fix', 'bbox_fix'],
        ),
    ),
]

visualizer = dict(name='visualizer', type='Mo2Cap2Visualizer', vis_backends=vis_backends)

work_dir = 'work_dirs/HMD_mo2cap2_smoke_test_2d_fix'
