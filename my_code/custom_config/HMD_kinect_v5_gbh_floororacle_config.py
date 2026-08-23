"""Task 41-D — oracle-T attribution at the TRAINING level (kinect).
DIAGNOSTIC-ONLY: canon_mode='floor' = the historical banned canon
w2f@c2w = relpose_to_floor(m2w) @ cam2middle (true extrinsic, translation
included) via temporal_cam2world labels — same oracle status as E2'.
Everything else identical to HMD_kinect_v5_gbh_wcalib_config.py (41-B):
same E2' attention arm, same SensorV3 assembly, same regime — only the
canon source changes.

Launch (dist 2-GPU):
    bash tools/dist_train.sh \
        my_code/custom_config/HMD_kinect_v5_gbh_floororacle_config.py 2 \
        --work-dir work_dirs/t41_floororacle
"""

import platform

assert platform.system() != 'Windows', 'server only'

custom_imports = dict(
    imports=['my_code.custom_config.ours_a_modules',
             'my_code.custom_config.ours_ec_modules'],
    allow_failed_imports=False,
)

# ---- data ----
data_root_train = '/mnt/dataset_vol/kinect_v5_split/Train'
data_root_test = '/mnt/dataset_vol/kinect_v5_split/Val'
pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
frame_export_root = '/mnt/dataset_vol/frame_export'
kp2d_cache_name = 'kp2d_ours_pilot'

load_from = None            # E2'/P2 gate regime: COCO backbone init only

# ---- training (gate recipe: 10ep AdamW 5e-4 [5,8]) ----
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
_max_epochs = 10

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epochs,
                 val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(optimizer=dict(lr=0.0005, type='AdamW'))
param_scheduler = [dict(type='MultiStepLR', begin=0, end=_max_epochs,
                        milestones=[5, 8], gamma=0.5, by_epoch=True)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='xregopose/Full Body_All_mpjpe', rule='less'),
    visualization=dict(enable=False, type='PoseVisualizationHook'),
)

randomness = dict(seed=42, deterministic=False)
resume = False

codec = dict(type='OursSensorV3Codec', heatmap_size=(47, 47),
             input_size=(256, 256), sigma=3)

custom_hooks = [dict(type='SyncBuffersHook')]

model_wrapper_cfg = dict(type='MMDistributedDataParallel',
                         find_unused_parameters=True)
default_scope = 'mmpose'
env_cfg = dict(cudnn_benchmark=False, dist_cfg=dict(backend='nccl'),
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=6, type='LogProcessor',
                     window_size=50)

# ---- model ----
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(depth=101, type='ResNet',
                  init_cfg=dict(checkpoint=pretrained_coco, type='Pretrained')),
    data_preprocessor=dict(bgr_to_rgb=True, mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           type='PoseDataPreprocessor'),
    head=dict(
        type='OursAttnCascadedHead',
        attn_mode='replace',
        use_canon=True,
        canon_mode='floor',
        pe_mode='fixed',
        sensor_mode='off',
        spatial_in_query=True,
        heatmap_decoder_type='efficient',
        hmd_info_size=12,
        in_channels=2048, out_channels=16, decoder=codec,
        loss=dict(loss_weight=1000, type='KeypointMSELoss',
                  use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(loss_weight=500, type='KeypointMSELoss',
                                use_target_weight=False),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_hmd=dict(type='MSELoss'),
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss'),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss'),
    ),
    test_cfg=dict(flip_test=False, output_heatmaps=False),
)

# ---- pipeline / data (t26 SensorV3 assembly) ----
_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
    'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
    'flip_direction', 'flip_indices', 'raw_ann_info', 'dataset_name', 'action',
)
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]
val_pipeline = train_pipeline

data_mode = 'topdown'
dataset_type = 'KinectEgoposeSensorV3Dataset'
_ds_common = dict(
    type=dataset_type, data_mode=data_mode, frame_export_root=frame_export_root,
    kp2d_cache_name=kp2d_cache_name, history=1,
    ground_info_mode='both_from_ground',
    filter_cfg=dict(filter_empty_gt=False, min_size=32), use_2d_visible=True,
    sensors_file='sensors_v3.npz',
)
dataset_train = dict(**_ds_common, data_root=data_root_train,
                     pipeline=train_pipeline)
dataset_val = dict(**_ds_common, data_root=data_root_test,
                   pipeline=val_pipeline, test_mode=True)

_num_workers = 4
train_dataloader = dict(
    batch_size=16, dataset=dataset_train, drop_last=True,
    num_workers=_num_workers, persistent_workers=True, pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_dataloader = dict(
    batch_size=16, dataset=dataset_val, drop_last=False,
    num_workers=_num_workers, persistent_workers=True, pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=None, type='CustomxRegoposeMetric',
                     use_action=True)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(name='visualizer', vis_backends=vis_backends,
                  type='PoseLocalVisualizer')

work_dir = 'work_dirs/t41_floororacle'
