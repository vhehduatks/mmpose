auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
codec = dict(
    input_size=(
        640,
        640,
    ), type='Egoposecodec')
custom_hooks = [
    dict(
        new_train_pipeline=[
            dict(type='LoadImage'),
            dict(
                bbox_keep_corner=False,
                clip_border=True,
                input_size=(
                    640,
                    640,
                ),
                pad_val=(
                    114,
                    114,
                    114,
                ),
                rotate_prob=0,
                scale_prob=0,
                scale_type='long',
                shift_prob=0,
                type='BottomupRandomAffine'),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip'),
            dict(
                by_box=True,
                by_kpt=True,
                keep_empty=False,
                type='FilterAnnotations'),
            dict(
                encoder=dict(input_size=(
                    640,
                    640,
                ), type='Egoposecodec'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        num_last_epochs=20,
        priority=48,
        type='YOLOXPoseModeSwitchHook'),
    dict(priority=48, type='SyncNormHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_root = 'C:\\Users\\user\\Documents\\GitHub\\mmpose\\data'
dataset_custom = dict(
    ann_file='F:\\dataset_test',
    data_root='C:\\Users\\user\\Documents\\GitHub\\mmpose\\data',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    indices=range(0, 300),
    pipeline=[
        dict(backend_args=None, type='LoadImage'),
        dict(
            by_box=False,
            by_kpt=True,
            keep_empty=False,
            type='EgoposeFilterAnnotations'),
        dict(type='FisheyeCropTransform'),
        dict(
            encoder=dict(input_size=(
                640,
                640,
            ), type='Egoposecodec'),
            type='GenerateTarget'),
        dict(type='PackPoseInputs'),
    ],
    type='CustomEgoposeDataset')
deepen_factor = 0.33
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=True, type='PoseVisualizationHook')
	)
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    640,
    640,
)
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(type='Swish'),
        deepen_factor=0.33,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='CSPDarknet',
        widen_factor=0.5),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=1,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type='BatchSyncRandomResize'),
        ],
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            1,
            1,
            1,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        assigner=dict(dynamic_k_indicator='oks', type='SimOTAAssigner'),
        featmap_strides=(
            8,
            16,
            32,
        ),
        head_module_cfg=dict(
            act_cfg=dict(type='Swish'),
            feat_channels=256,
            in_channels=256,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            stacked_convs=2,
            widen_factor=0.5),
        loss_mpjpe=dict(
            loss_weight=1.0, type='MPJPELoss', use_target_weight=False),
        num_keypoints=15,
        overlaps_power=0.5,
        prior_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        type='YOLOXPoseHead_Custom'),
    init_cfg=dict(
        a=2.23606797749979,
        distribution='uniform',
        layer='Conv2d',
        mode='fan_in',
        nonlinearity='leaky_relu',
        type='Kaiming'),
    neck=dict(
        act_cfg=dict(type='Swish'),
        in_channels=[
            128,
            256,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=1,
        out_channels=128,
        type='YOLOXPAFPN',
        upsample_cfg=dict(mode='nearest', scale_factor=2),
        use_depthwise=False),
    test_cfg=dict(nms_thr=0.65, score_thr=0.01),
    type='BottomupPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        type='QuadraticWarmupLR'),
    dict(
        T_max=280,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=280,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
    dict(begin=280, by_epoch=True, end=300, factor=1, type='ConstantLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='F:\\dataset_test',
        data_root='C:\\Users\\user\\Documents\\GitHub\\mmpose\\data',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(type='LoadImage'),
            dict(
                by_box=False,
                by_kpt=True,
                keep_empty=False,
                type='EgoposeFilterAnnotations'),
            dict(type='FisheyeCropTransform'),
            dict(
                encoder=dict(input_size=(
                    640,
                    640,
                ), type='Egoposecodec'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CustomEgoposeDataset'),
    drop_last=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    collect_device='cpu', mode='mpjpe', prefix=[], skip_list=[], type='MPJPE')
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=5,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='F:\\dataset_test',
        data_root='C:\\Users\\user\\Documents\\GitHub\\mmpose\\data',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImage'),
            dict(
                by_box=False,
                by_kpt=True,
                keep_empty=False,
                type='EgoposeFilterAnnotations'),
            dict(type='FisheyeCropTransform'),
            dict(
                encoder=dict(input_size=(
                    640,
                    640,
                ), type='Egoposecodec'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CustomEgoposeDataset'),
    drop_last=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImage'),
    dict(
        by_box=False,
        by_kpt=True,
        keep_empty=False,
        type='EgoposeFilterAnnotations'),
    dict(type='FisheyeCropTransform'),
    dict(
        encoder=dict(input_size=(
            640,
            640,
        ), type='Egoposecodec'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        bbox_keep_corner=False,
        clip_border=True,
        input_size=(
            640,
            640,
        ),
        pad_val=(
            114,
            114,
            114,
        ),
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        shift_prob=0,
        type='BottomupRandomAffine'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(by_box=True, by_kpt=True, keep_empty=False, type='FilterAnnotations'),
    dict(
        encoder=dict(input_size=(
            640,
            640,
        ), type='Egoposecodec'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='F:\\dataset_test',
        data_root='C:\\Users\\user\\Documents\\GitHub\\mmpose\\data',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(type='LoadImage'),
            dict(
                by_box=False,
                by_kpt=True,
                keep_empty=False,
                type='EgoposeFilterAnnotations'),
            dict(type='FisheyeCropTransform'),
            dict(
                encoder=dict(input_size=(
                    640,
                    640,
                ), type='Egoposecodec'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CustomEgoposeDataset'),
    drop_last=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    collect_device='cpu', mode='mpjpe', prefix=[], skip_list=[], type='MPJPE')
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        by_box=False,
        by_kpt=True,
        keep_empty=False,
        type='EgoposeFilterAnnotations'),
    dict(type='FisheyeCropTransform'),
    dict(
        encoder=dict(input_size=(
            640,
            640,
        ), type='Egoposecodec'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Pose3dLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
widen_factor = 0.5
work_dir = 'temp_workdir'
