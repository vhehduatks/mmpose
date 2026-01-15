ann_file_test = '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TestSet'
ann_file_train = '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TrainSet'
ann_file_val = '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/ValSet'
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
coco_pretrained_resnet101_256x192 = '/home/jovyan/vol_arvr_hyeonghwan/mmpose/coco_pose_resnet_101_256x192.pth.tar'
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
    data_root='/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TrainSet',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
    type='CustomEgoposeDataset')
dataset_mo2cap2_val = dict(
    data_mode='topdown',
    data_root='/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TestSet',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
    type='CustomEgoposeDataset')
dataset_type = 'CustomEgoposeDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=2,
        max_keep_ckpts=3,
        rule='less',
        save_best='xregopose/Full Body_All_mpjpe',
        type='CheckpointHook'),
    visualization=dict(
        enable=True, interval=3, kpt_thr=0.3, type='PoseVisualizationHook'))
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
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=
            '/home/jovyan/vol_arvr_hyeonghwan/mmpose/coco_pose_resnet_101_256x192.pth.tar',
            type='Pretrained'),
        type='ResNet'),
    backbone2=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=
            '/home/jovyan/vol_arvr_hyeonghwan/mmpose/pose_resnet_101_256x256.pth.tar',
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
            loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_backbone_latant=dict(loss_weight=1.0, type='MSELoss'),
        loss_cosine_similarity=dict(loss_weight=1.0, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=500, type='KeypointMSELoss', use_target_weight=False),
        loss_hmd=dict(type='MSELoss'),
        loss_limb_length=dict(loss_weight=1.0, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        out_channels=16,
        type='CustomxRegoposeBaselinel1_multi_backbone'),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False),
    type='Custom_TopdownPoseEstimator')
mpii_pretrained_resnet101_256x256 = '/home/jovyan/vol_arvr_hyeonghwan/mmpose/pose_resnet_101_256x256.pth.tar'
mpii_pretrained_resnet101_384x384 = '/home/jovyan/vol_arvr_hyeonghwan/mmpose/pose_resnet_101_384x384.pth.tar'
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW'),
    paramwise_cfg=dict(custom_keys=dict()))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=10,
        gamma=0.5,
        milestones=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        type='MultiStepLR'),
]
randomness = dict(diff_rank_seed=True, seed=42)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=58,
    dataset=dict(
        data_mode='topdown',
        data_root=
        '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TestSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
        type='CustomEgoposeDataset'),
    drop_last=False,
    num_workers=48,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=None, type='CustomxRegoposeMetric', use_action=True)
torchvision = 'torchvision://resnet101'
train_cfg = dict(max_epochs=10, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=58,
    dataset=dict(
        data_mode='topdown',
        data_root=
        '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TrainSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
        type='CustomEgoposeDataset'),
    drop_last=True,
    num_workers=48,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler'))
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
    batch_size=58,
    dataset=dict(
        data_mode='topdown',
        data_root=
        '/home/jovyan/vol_arvr_hyeonghwan2/xr_egopose_dataset/TestSet',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
        type='CustomEgoposeDataset'),
    drop_last=False,
    num_workers=48,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=None, type='CustomxRegoposeMetric', use_action=False)
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
        init_kwargs=dict(project='mmpose_xregopose_baseline_recall_test'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(project='mmpose_xregopose_baseline_recall_test'),
            type='WandbVisBackend'),
    ])
work_dir = 'work_dirs/xr_egopose_2backbone_lr_change'
