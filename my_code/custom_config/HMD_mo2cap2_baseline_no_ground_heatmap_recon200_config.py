"""
Mo2Cap2 Baseline - NO Ground Reference - Heatmap Recon Loss Weight Ablation

Testing hypothesis: Previous experiment's lower heatmap recon weight vs current.

Change from baseline_no_ground:
  - loss_heatmap_recon: loss_weight 500 -> 200

Previous experiment: loss_heatmap_recon weight = 200
Current experiment: loss_heatmap_recon weight = 500
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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
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
        end=10,
        milestones=[6, 8],
        gamma=0.1,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='mo2cap2/Full Body_All_mpjpe',
        rule='less',
        max_keep_ckpts=3,
    ),
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,
        interval=35,
        train_interval=3300,
    ),
    logger=dict(type='LoggerHook', interval=50),
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

# ABLATION: loss_heatmap_recon weight changed from 500 to 200
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
        type='CustomMo2Cap2BaselineHead',
        in_channels=2048,
        out_channels=15,
        hmd_info_size=9,
        heatmap_decoder_type='efficient',
        decoder=codec,
        loss=dict(loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity', skeleton=MO2CAP2_SKELETON),
        loss_heatmap_recon=dict(loss_weight=200, type='KeypointMSELoss', use_target_weight=False),  # Changed from 500
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
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

train_pipeline = [
    dict(type='LoadImageFromH5'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_train_meta_keys),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='Mo2Cap2CenterCrop', margin_left=128, margin_right=128),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_val_meta_keys),
]

data_mode = 'topdown'

dataset_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=train_pipeline,
    input_size=(256, 256),
    sample_interval=1,
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
    _num_workers = 8
    _persistent_workers = True
    _batch_size = 64

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

test_dataloader = val_dataloader
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mo2cap2-pose-estimation',
            name='baseline_no_ground_heatmap_recon200',
            tags=['mo2cap2', 'baseline', 'no_ground', 'loss_ablation'],
        ),
    ),
]

visualizer = dict(name='visualizer', type='Mo2Cap2Visualizer', vis_backends=vis_backends)

work_dir = 'work_dirs/HMD_mo2cap2_baseline_no_ground_heatmap_recon200'
