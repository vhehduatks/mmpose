"""
Mo2Cap2 Cascaded Refinement + Both From Ground V2b Config

Adapted from EgoPose cascaded_both_from_ground_v2b for Mo2Cap2 dataset.

Key Differences from EgoPose version:
  1. 15 joints (Mo2Cap2) vs 16 joints (EgoPose)
  2. Mo2Cap2 skeleton connections
  3. Mo2Cap2-specific HMD info computation (Neck=0, RightHand=3, LeftHand=6)
  4. Ground reference from Neck (as per user requirement)

Model Architecture:
  - Stage 1: ResNet101 -> Deconv -> Heatmap -> Encoder -> Z -> PoseDecoder -> Coarse3D
  - Stage 2: GridSample + KinFeatures + HMD -> RefinementMLP -> Refined3D

HMD Info (12-dim, both_from_ground mode):
  - Base HMD (9): right_local(3), left_local(3), distances(3)
  - Enhanced (3): neck_from_ground(1), left_hand_from_ground(1), right_hand_from_ground(1)

Training: 20 epochs with EfficientHeatmapDecoder
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

# V2b: Extended training (20 epochs)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1,
)
val_cfg = dict()
test_cfg = None

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW'),
)

# V2b: Adjusted milestones for 20 epochs
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        milestones=[8, 14],
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='less',
        save_best='mo2cap2/Full Body_All_mpjpe',
        type='CheckpointHook',
        by_epoch=True
    ),
    visualization=dict(
        enable=True,
        interval=100,
        kpt_thr=0.3,
        type='PoseVisualizationHook'
    )
)

randomness = dict(seed=42, deterministic=False)
resume = False

# Mo2Cap2 codec (15 joints, 47x47 heatmap)
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

# Model with Mo2Cap2-specific cascaded head
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
        type='CustomMo2Cap2CascadedRefinementHead_enhanced',
        in_channels=2048,
        out_channels=15,  # Mo2Cap2 has 15 joints
        hmd_info_size=12,  # 9 base + 3 (neck + left_hand + right_hand from ground)
        # V2b Optimization: EfficientHeatmapDecoder
        heatmap_decoder_type='efficient',
        decoder=codec,
        # Stage 1 losses (baseline)
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        # Stage 2 losses (refinement)
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss'),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss'),
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
    'h5_chunk_idx', 'h5_local_idx', 'h5_chunk_path'
)

# Pipeline with Mo2Cap2-specific EnhanceHMDInfo transform
train_pipeline = [
    dict(type='LoadImageFromH5'),  # Load from H5 chunk files
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5'),  # Use H5 for validation too
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

data_mode = 'topdown'

# Training dataset - H5 FAST loading
dataset_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=train_pipeline,
    input_size=(256, 256),
    sample_interval=1,  # Use all 530k samples
    use_zoom=False,     # Do not use zoomed images
)

# Validation dataset - Also use H5 for consistency
# Note: If test set is in different format, may need Mo2Cap2CocoDataset
dataset_val = dict(
    type='H5Mo2Cap2Dataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=val_pipeline,
    input_size=(256, 256),
    sample_interval=50,  # Sample every 50 for validation (10k samples)
    use_zoom=False,
    test_mode=True,
)

if IS_WINDOWS:
    _num_workers = 0
    _persistent_workers = False
    _batch_size = 48
else:
    _num_workers = 6
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

val_evaluator = dict(
    ann_file=None,
    type='CustomMo2Cap2Metric',
    use_action=False  # Training set doesn't have action labels
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(project='mmpose_mo2cap2_cascaded_both_from_ground_v2b'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_mo2cap2_cascaded_both_from_ground_v2b'
