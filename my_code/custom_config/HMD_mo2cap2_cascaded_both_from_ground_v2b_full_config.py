"""
Mo2Cap2 Cascaded Refinement + Both From Ground V2b - FULL TRAINING

Model: Cascaded Refinement with Enhanced HMD (12-dim: 9 base + 3 ground ref)
Data:
  - Training: H5 chunks from /mnt/sdb2/mo2cap2_dataset/training_data (~530k samples)
  - Validation: JPG + MAT from /mnt/sdb2/mo2cap2_dataset/test_data/TestSet (5646 samples)

Evaluation: Official MATLAB protocol (Procrustes without scaling, per-action breakdown)
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

# Training configuration (10 epochs)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=2,  # Validate every 2 epochs
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
        end=10,
        milestones=[6, 8],
        gamma=0.1,
        by_epoch=True
    ),
]

# Checkpoint saving
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,  # Save every 2 epochs
        save_best='mo2cap2/Full Body_All_mpjpe',
        rule='less',
        max_keep_ckpts=3,
    ),
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,  # Enable visualization during validation
        interval=50,  # Visualize every 50 samples
        # NOTE: Do NOT set out_dir - it disables WandB image logging
    ),
    logger=dict(type='LoggerHook', interval=50),
)

randomness = dict(seed=42, deterministic=False)
resume = False

# Mo2Cap2 skeleton (14 bones for 15 joints)
MO2CAP2_SKELETON = [
    (0, 1),   # Neck -> RightArm
    (1, 2),   # RightArm -> RightForeArm
    (2, 3),   # RightForeArm -> RightHand
    (0, 4),   # Neck -> LeftArm
    (4, 5),   # LeftArm -> LeftForeArm
    (5, 6),   # LeftForeArm -> LeftHand
    (0, 7),   # Neck -> RightUpLeg
    (7, 8),   # RightUpLeg -> RightLeg
    (8, 9),   # RightLeg -> RightFoot
    (9, 10),  # RightFoot -> RightToeBase
    (0, 11),  # Neck -> LeftUpLeg
    (11, 12), # LeftUpLeg -> LeftLeg
    (12, 13), # LeftLeg -> LeftFoot
    (13, 14), # LeftFoot -> LeftToeBase
]

# Mo2Cap2 symmetric limbs: ((left_parent, left_child), (right_parent, right_child))
MO2CAP2_SYMMETRIC_LIMBS = [
    ((0, 4), (0, 1)),     # Neck->LeftArm, Neck->RightArm
    ((4, 5), (1, 2)),     # LeftArm->LeftForeArm, RightArm->RightForeArm
    ((5, 6), (2, 3)),     # LeftForeArm->LeftHand, RightForeArm->RightHand
    ((0, 11), (0, 7)),    # Neck->LeftUpLeg, Neck->RightUpLeg
    ((11, 12), (7, 8)),   # LeftUpLeg->LeftLeg, RightUpLeg->RightLeg
    ((12, 13), (8, 9)),   # LeftLeg->LeftFoot, RightLeg->RightFoot
    ((13, 14), (9, 10)),  # LeftFoot->LeftToeBase, RightFoot->RightToeBase
]

# Mo2Cap2 codec (15 joints, 47x47 heatmap)
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

# DDP config for distributed training (handle unused parameters)
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
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
        heatmap_decoder_type='efficient',
        decoder=codec,
        # Stage 1 losses
        loss=dict(
            loss_weight=1000,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity', skeleton=MO2CAP2_SKELETON),
        loss_heatmap_recon=dict(
            loss_weight=500,
            type='KeypointMSELoss',
            use_target_weight=False
        ),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        # Stage 2 losses
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss', skeleton=MO2CAP2_SKELETON),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss', symmetric_limbs=MO2CAP2_SYMMETRIC_LIMBS),
    ),
    test_cfg=dict(
        flip_test=False,
        output_heatmaps=False
    ),
)

# Meta keys for training (includes H5 fields)
_train_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'h5_chunk_idx', 'h5_local_idx', 'h5_chunk_path'
)

# Meta keys for validation (includes frame_idx/sequence_name for official eval)
_val_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'frame_idx', 'sequence_name'  # For official per-action evaluation
)

# Training pipeline: Load from H5 chunks
train_pipeline = [
    dict(type='LoadImageFromH5'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_train_meta_keys),
]

# Validation pipeline: Load JPG images from disk
# Mo2Cap2 test images: 1280x1024, crop 128px margins from each horizontal side -> 1024x1024
val_pipeline = [
    dict(type='LoadImage'),  # Load from JPG files
    dict(type='Mo2Cap2CenterCrop', margin_left=128, margin_right=128),  # Crop to 1024x1024
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo_Mo2Cap2', mode='both_from_ground'),
    dict(type='PackPoseInputs', meta_keys=_val_meta_keys),
]

data_mode = 'topdown'

# Training dataset: H5 chunks (full dataset, no sampling)
dataset_train = dict(
    type='H5Mo2Cap2Dataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=train_pipeline,
    input_size=(256, 256),
    sample_interval=1,  # Use all samples
    use_zoom=False,
)

# Validation dataset: Test set with JPG images + MAT ground truth
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

val_evaluator = dict(
    ann_file=None,
    type='CustomMo2Cap2Metric',
    use_action=True  # Enable per-action breakdown
)

# Visualization backends (WandB enabled)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mo2cap2-pose-estimation',
            name='cascaded_v2b_ground_ref_full',
            tags=['mo2cap2', 'cascaded', 'ground_ref', 'official_eval'],
        ),
    ),
]

visualizer = dict(
    name='visualizer',
    type='Mo2Cap2Visualizer',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_mo2cap2_cascaded_both_from_ground_v2b_full'
