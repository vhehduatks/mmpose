"""
ViT Lifting v6 - Extended 20 Epochs (Full Training Config)

Change from v6 (10ep):
    - max_epochs: 10 -> 20
    - CosineAnnealingLR T_max: 10 -> 20
    - Everything else identical

Rationale:
    v6 (10ep) was still improving at epoch 10 (45.94 -> 45.70mm),
    with LR at 1.32e-5. The CosineAnnealingLR reached near-zero by
    epoch 10, meaning the model barely finished converging.
    Extending to 20 epochs gives a longer high-LR phase and more
    gradual decay, allowing the reduced-capacity model (embed_dim=128,
    2 layers) more time to converge.

    Expected LR at key epochs:
        Epoch 5:  ~4.05e-4  (vs 3.28e-4 in 10ep)
        Epoch 10: ~2.50e-4  (vs 1.32e-5 in 10ep)
        Epoch 15: ~6.55e-5
        Epoch 20: ~1.00e-6

Architecture (unchanged from v6):
    Backbone feat [2048, 8, 8]
           |
    SPT (Shifted Patch Tokenization)
           |
    Depth-wise Conv Embedding
           |
    Spatial Tokens [64, 128] + Joint Queries [16, 128]
           |
    LSA (Locality Self-Attention) x 2
           |
    Joint Tokens [16, 128]
           |-- Heatmap Decoder (Reconstruction)
           |
    HMD Cross-Attention
           |
    3D Pose [16, 3]
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
    max_epochs=20,
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
# LR Schedule - CosineAnnealingLR with warmup (extended to 20 epochs)
# =============================================================================
param_scheduler = [
    # Warmup (500 iterations)
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    # CosineAnnealingLR (T_max=20)
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=20,
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
# Model - ViT Lifting v6 (Small Dataset Optimized) - unchanged
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
        type='CustomEgoposeViTLiftingHeadV6',
        in_channels=2048,
        out_channels=16,
        decoder=codec,
        # v6 small model params (reduced for small dataset)
        embed_dim=128,          # Reduced from 256
        num_heads=4,            # Reduced from 8
        num_layers=2,           # Reduced from 4
        mlp_ratio=2.0,          # Reduced from 4.0
        dropout=0.2,            # Increased from 0.1
        heatmap_size=47,
        init_temperature=0.5,   # LSA learnable temperature init
        # v6 specific flags
        use_spt=True,           # Shifted Patch Tokenization
        use_dwconv=True,        # Depth-wise Conv Embedding
        use_lsa=True,           # Locality Self-Attention
        use_hmd=True,
        use_heatmap_recon=True,
        # Losses
        loss_heatmap_recon=dict(
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
        init_kwargs=dict(project='mmpose_xregopose_vit_lifting_v6_20ep'),
        type='WandbVisBackend'
    ),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
)

work_dir = 'work_dirs/HMD_xregopose_vit_lifting_v6_20ep_full'
