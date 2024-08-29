ann_file_test = 'F:\\mo2cap2_data_half\\TestSet'
ann_file_train = 'F:\\mo2cap2_data_half\\TrainSet'
ann_file_val = 'F:\\mo2cap2_data_half\\ValSet'
# ---
# ann_file_test = r'F:\mo2cap2_data_small\TestSet'
# ann_file_val = r'F:\mo2cap2_data_small\ValSet'
# ann_file_train = r'F:\mo2cap2_data_small\TrainSet'

auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
coco_pretrained_resnet101_256x192 = 'C:\\Users\\user\\Downloads\\pytorch-20240821T053436Z-001\\pytorch\\pose_coco\\coco_pose_resnet_101_256x192.pth.tar'
mpii_pretrained_resnet101_256x256 = 'C:\\Users\\user\\.cache\\torch\\hub\\checkpoints\\pose_resnet_101_256x256.pth.tar'
mpii_pretrained_resnet101_384x384 = 'C:\\Users\\user\\Downloads\\pose_mpii\\pose_resnet_101_384x384.pth.tar'

codec = dict(
    heatmap_size=(
        47,
        47,
    ),
    input_size=(
        256,
        256,
    ),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
dataset_mo2cap2_train = dict(
    data_mode='topdown',
    data_root='F:\\mo2cap2_data_half\\TrainSet',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    input_size=(
        256,
        256,
    ),
    pipeline=[
        dict(type='LoadImage'),
        dict(padding=1.0, type='GetBBoxCenterScale'),
        dict(input_size=(
            256,
            256,
        ), type='TopdownAffine'),
        dict(
            encoder=dict(
                heatmap_size=(
                    47,
                    47,
                ),
                input_size=(
                    256,
                    256,
                ),
                sigma=3,
                type='Custom_mo2cap2_MSRAHeatmap'),
            type='GenerateTarget'),
        dict(type='PackPoseInputs'),
    ],
    type='Mo2Cap2CocoDataset')
dataset_mo2cap2_val = dict(
    data_mode='topdown',
    data_root='F:\\mo2cap2_data_half\\TestSet',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    input_size=(
        256,
        256,
    ),
    pipeline=[
        dict(type='LoadImage'),
        dict(padding=1.0, type='GetBBoxCenterScale'),
        dict(input_size=(
            256,
            256,
        ), type='TopdownAffine'),
        dict(
            encoder=dict(
                heatmap_size=(
                    47,
                    47,
                ),
                input_size=(
                    256,
                    256,
                ),
                sigma=3,
                type='Custom_mo2cap2_MSRAHeatmap'),
            type='GenerateTarget'),
        dict(type='PackPoseInputs'),
    ],
    test_mode=True,
    type='Mo2Cap2CocoDataset')
dataset_type = 'Mo2Cap2CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1000,
        max_keep_ckpts=3,
        rule='less',
        save_best='mo2cap2/Full Body_All_mpjpe',
        type='CheckpointHook'),
    visualization=dict(
        enable=True, interval=15, kpt_thr=0.3, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)


model = dict(
	type='Custom_TopdownPoseEstimator',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=coco_pretrained_resnet101_256x192,
            type='Pretrained'),
        type='ResNet'),
	backbone2=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=mpii_pretrained_resnet101_256x256,
            type='Pretrained'),
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                47,
                47,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'),
        in_channels=2048,
        loss=dict(
            loss_weight=1000, type='KeypointMSELoss', use_target_weight=True),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=500, type='KeypointMSELoss', use_target_weight=True),
        loss_limb_length=dict(loss_weight=0.5, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        out_channels=15,
        type='CustomMo2Cap2Baselinel1_multi_backbone'),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False),
    )


optim_wrapper = dict(
	optimizer=dict(lr=0.0005, type='AdamW'),
	paramwise_cfg=dict(
	custom_keys={
		'backbone2': dict(lr_mult=0)
	}
	)
)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=70000,
        gamma=0.5,
        milestones=[
            5000,
            10000,
            15000,
            20000,
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
            55000,
            60000,
            65000,
            70000,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=42)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_mode='topdown',
        data_root='F:\\mo2cap2_data_half\\TestSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        input_size=(
            256,
            256,
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        47,
                        47,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=3,
                    type='Custom_mo2cap2_MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='Mo2Cap2CocoDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=None, type='CustomMo2Cap2Metric', use_action=True)
torchvision = 'torchvision://resnet101'
train_cfg = dict(
    max_iters=100000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_mode='topdown',
        data_root='F:\\mo2cap2_data_half\\TrainSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        input_size=(
            256,
            256,
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        47,
                        47,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=3,
                    type='Custom_mo2cap2_MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='Mo2Cap2CocoDataset'),
    drop_last=True,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=True, shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                47,
                47,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_mode='topdown',
        data_root='F:\\mo2cap2_data_half\\TestSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        input_size=(
            256,
            256,
        ),
        pipeline=[
            dict(type='LoadImage'),
            dict(padding=1.0, type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        47,
                        47,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=3,
                    type='Custom_mo2cap2_MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='Mo2Cap2CocoDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=None, type='CustomMo2Cap2Metric', use_action=False)
val_pipeline = [
    dict(type='LoadImage'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                47,
                47,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=3,
            type='Custom_mo2cap2_MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(project='mmpose_mo2cap2_baseline_middle'),
        type='WandbVisBackend'),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer',
    vis_backends=vis_backends
	)
work_dir = 'work_dirs/HMD_mo2cap2_test'
