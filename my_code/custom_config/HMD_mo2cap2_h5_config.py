"""
H5 Mo2Cap2 Fast Loading Config

This config uses H5Mo2Cap2Dataset which loads data directly from the original
mo2cap2 HDF5 chunk files. No preprocessing required - just point to the data directory.

Performance improvement:
    - Original (JSON parsing): ~5-10 minutes to load 530k samples
    - H5 Direct: ~3-5 seconds to load 530k samples

Usage:
    python tools/train.py my_code/custom_config/HMD_mo2cap2_h5_config.py
"""

_base_ = r'C:\Users\user\Documents\GitHub\mmpose\configs\_base_\default_runtime.py'

# ======================= Runtime Config =======================
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=100000,
    val_interval=1000
)
auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        save_best='mo2cap2/Full Body_All_mpjpe',
        rule='less',
        max_keep_ckpts=3
    ),
    visualization=dict(type='PoseVisualizationHook', enable=True, interval=15, kpt_thr=0.3),
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=70000,
        milestones=[5000 * i for i in range(1, 15)],
        gamma=0.5,
        by_epoch=False
    )
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
    )
)

# ======================= Model Config =======================
codec = dict(
    type='Custom_mo2cap2_MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(47, 47),
    sigma=3
)

# Pretrained weights
coco_pretrained_resnet101_256x192 = r'C:\Users\user\Downloads\pytorch-20240821T053436Z-001\pytorch\pose_coco\coco_pose_resnet_101_256x192.pth.tar'

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint=coco_pretrained_resnet101_256x192),
    ),
    head=dict(
        type='CustomMo2Cap2Baselinel1',
        in_channels=2048,
        out_channels=15,
        loss=dict(type='KeypointMSELoss', use_target_weight=True, loss_weight=1000),
        loss_pose_l2norm=dict(type='pose_l2norm', loss_weight=1.),
        loss_cosine_similarity=dict(type='cosine_similarity', loss_weight=0.1),
        loss_limb_length=dict(type='limb_length', loss_weight=0.5),
        loss_heatmap_recon=dict(type='KeypointMSELoss', use_target_weight=True, loss_weight=500),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        output_heatmaps=True,
    )
)

# ======================= Dataset Config =======================
data_mode = 'topdown'

# Data paths
# Training: uses HDF5 chunks (530 files x 1000 samples = 530k samples)
h5_data_root_train = r'F:\mo2cap2_dataset\training_data'

# Test: uses folder structure (different format from training)
# Test set has: olek_outdoor/, weipeng_studio/ with rgba/json subfolders
test_data_root = r'F:\mo2cap2_dataset\test_data\TestSet'

# ------------------- Training Pipeline (H5) -------------------
train_pipeline = [
    dict(type='LoadImageFromH5'),  # Load from H5 chunk files
    dict(type='GetBBoxCenterScale', padding=1.),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# ------------------- Validation Pipeline (File-based) -------------------
val_pipeline = [
    dict(type='LoadImage'),  # Standard file loading for test set
    dict(type='GetBBoxCenterScale', padding=1.),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# Training dataset - H5 FAST loading (~530k samples in ~3-5 seconds)
dataset_mo2cap2_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root=h5_data_root_train,
    data_mode=data_mode,
    pipeline=train_pipeline,
    input_size=codec['input_size'],
    sample_interval=1,  # Use all samples (set to 10 for quick testing)
    use_zoom=False,  # Set True to use ZoomImages
)

# Validation dataset - uses original Mo2Cap2CocoDataset for test set
# (test set has different format: folder structure with JSON files)
dataset_mo2cap2_val = dict(
    type='Mo2Cap2CocoDataset',
    data_root=test_data_root,
    data_mode=data_mode,
    pipeline=val_pipeline,
    test_mode=True,
    input_size=codec['input_size'],
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
)

# ======================= DataLoader Config =======================
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,  # Keep workers alive for faster epoch starts
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=True),
    dataset=dataset_mo2cap2_train
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dataset_mo2cap2_val
)

test_dataloader = val_dataloader

# ======================= Evaluator Config =======================
val_evaluator = dict(
    type='CustomMo2Cap2Metric',
    ann_file=None,
    use_action=False,
)
test_evaluator = dict(
    type='CustomMo2Cap2Metric',
    ann_file=None,
    use_action=True,
)

# ======================= Visualizer Config =======================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project="mmpose_mo2cap2_h5_fast",
        ),
    ),
]
visualizer = dict(
    type='CustomPose3dLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
