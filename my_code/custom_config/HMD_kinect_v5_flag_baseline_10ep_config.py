"""
egodataset_flag_fixed_ver5 - Single-Stage + Baseline 9-dim (10 epochs)
- Dataset: KinectEgoposeDataset (egodataset_flag_fixed_ver5)
- Split: 32 train batches (632 sessions) / 9 val batches (179 sessions) ~80/20
- Backbone: ResNet-101 pretrained on COCO
- Head: CustomxRegoposeBaselinel1 (hmd_info_size=9)
- Epochs: 10
"""

import platform

IS_WINDOWS = platform.system() == 'Windows'

# =============================================================================
# Data paths
# =============================================================================
if IS_WINDOWS:
    data_root_train = r'C:\placeholder\Train'
    data_root_test = r'C:\placeholder\Test'
    pretrained_coco = r'F:\egodataset_cache\pose_coco\coco_pose_resnet_101_256x192.pth.tar'
else:
    data_root_train = '/mnt/dataset_vol/kinect_v5_split/Train'
    data_root_test = '/mnt/dataset_vol/kinect_v5_split/Val'
    pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'

# =============================================================================
# Training Config
# =============================================================================
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')

_max_epochs = 10

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
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
        end=_max_epochs,
        milestones=[5, 8],
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='xregopose/Full Body_All_mpjpe',
        rule='less',
    ),
    visualization=dict(
        enable=True,
        interval=50,
        kpt_thr=0.3,
        type='PoseVisualizationHook'
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
# Model
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
        out_channels=16,
        type='CustomxRegoposeBaselinel1'
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
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

# =============================================================================
# Dataset Config
# =============================================================================
data_mode = 'topdown'
dataset_type = 'KinectEgoposeDataset'

dataset_train = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root_train,
    ground_info_mode=None,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    use_2d_visible=True,
    pipeline=train_pipeline,
)

dataset_val = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root_test,
    ground_info_mode=None,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    use_2d_visible=True,
    pipeline=val_pipeline,
    test_mode=True,
)

# =============================================================================
# DataLoader Config
# =============================================================================
if IS_WINDOWS:
    _num_workers = 0
    _persistent_workers = False
    _batch_size = 8
else:
    _num_workers = 4
    _persistent_workers = True
    _batch_size = 16

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
# Visualization
# =============================================================================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='egodataset_flag_fixed_ver5-baseline-9dim',
            name='HMD_kinect_v5_flag_baseline_10ep',
            tags=['kinect_v5_flag', 'baseline', '10ep'],
        ),
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_kinect_v5_flag_baseline_10ep'
