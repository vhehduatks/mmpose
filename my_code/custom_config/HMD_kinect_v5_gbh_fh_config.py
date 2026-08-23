"""Task 39 E2' (GT-body-axis canon gate) (kinect) — derived from HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py; changes: head arm params + work_dir only.
"""
custom_imports = dict(
    imports=['my_code.custom_config.ours_a_modules'],
    allow_failed_imports=False,
)
IS_WINDOWS = False
_batch_size = 16
_max_epochs = 10
_meta_keys = (
    'id',
    'img_id',
    'img_path',
    'category_id',
    'crowd_index',
    'ori_shape',
    'img_shape',
    'input_size',
    'input_center',
    'input_scale',
    'flip',
    'flip_direction',
    'flip_indices',
    'raw_ann_info',
    'dataset_name',
    'action',
)
_num_workers = 4
_persistent_workers = True
auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
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
data_root_test = '/mnt/dataset_vol/kinect_v5_split/Val'
data_root_train = '/mnt/dataset_vol/kinect_v5_split/Train'
dataset_train = dict(
    data_mode='topdown',
    data_root='/mnt/dataset_vol/kinect_v5_split/Train',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ground_info_mode='both_from_ground',
    pipeline=[
        dict(type='LoadImage'),
        dict(input_size=(
            256,
            256,
        ), type='EgoImageResize'),
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
        dict(
            meta_keys=(
                'id',
                'img_id',
                'img_path',
                'category_id',
                'crowd_index',
                'ori_shape',
                'img_shape',
                'input_size',
                'input_center',
                'input_scale',
                'flip',
                'flip_direction',
                'flip_indices',
                'raw_ann_info',
                'dataset_name',
                'action',
            ),
            type='PackPoseInputs'),
    ],
    type='KinectEgoposeDataset',
    use_2d_visible=True)
dataset_type = 'KinectEgoposeDataset'
dataset_val = dict(
    data_mode='topdown',
    data_root='/mnt/dataset_vol/kinect_v5_split/Val',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ground_info_mode='both_from_ground',
    pipeline=[
        dict(type='LoadImage'),
        dict(input_size=(
            256,
            256,
        ), type='EgoImageResize'),
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
        dict(
            meta_keys=(
                'id',
                'img_id',
                'img_path',
                'category_id',
                'crowd_index',
                'ori_shape',
                'img_shape',
                'input_size',
                'input_center',
                'input_scale',
                'flip',
                'flip_direction',
                'flip_indices',
                'raw_ann_info',
                'dataset_name',
                'action',
            ),
            type='PackPoseInputs'),
    ],
    test_mode=True,
    type='KinectEgoposeDataset',
    use_2d_visible=True)
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='less',
        save_best='xregopose/Full Body_All_mpjpe',
        type='CheckpointHook'),
    visualization=dict(
        enable=True, interval=50, kpt_thr=0.3, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=
            '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar',
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
        heatmap_decoder_type='efficient',
        hmd_info_size=12,
        in_channels=2048,
        loss=dict(
            loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_bone_length=dict(loss_weight=0.5, type='bone_length_loss'),
        loss_cosine_similarity=dict(loss_weight=0.1, type='cosine_similarity'),
        loss_heatmap_recon=dict(
            loss_weight=500, type='KeypointMSELoss', use_target_weight=False),
        loss_hmd=dict(type='MSELoss'),
        loss_limb_length=dict(loss_weight=0.25, type='limb_length'),
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_pose_l2norm_refined=dict(loss_weight=1.0, type='pose_l2norm'),
        loss_symmetry=dict(loss_weight=0.1, type='symmetry_loss'),
        out_channels=16,
        attn_mode='replace',
        use_canon=True,
        canon_mode='frame_head',
        frame_joints=(1, 4, 7),
        pe_mode='fixed',
        sensor_mode='off',
        spatial_in_query=True,
        type='OursAttnCascadedHead'),
    test_cfg=dict(flip_test=False, output_heatmaps=False),
    type='TopdownPoseEstimator')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
optim_wrapper = dict(optimizer=dict(lr=0.0005, type='AdamW'))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=10,
        gamma=0.5,
        milestones=[
            5,
            8,
        ],
        type='MultiStepLR'),
]
pretrained_coco = '/mnt/dataset_vol/pretrained/coco_pose_resnet_101_256x192.pth.tar'
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = None
train_cfg = dict(max_epochs=10, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_mode='topdown',
        data_root='/mnt/dataset_vol/kinect_v5_split/Train',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ground_info_mode='both_from_ground',
        pipeline=[
            dict(type='LoadImage'),
            dict(input_size=(
                256,
                256,
            ), type='EgoImageResize'),
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
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'category_id',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'dataset_name',
                    'action',
                ),
                type='PackPoseInputs'),
        ],
        type='KinectEgoposeDataset',
        use_2d_visible=True),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(input_size=(
        256,
        256,
    ), type='EgoImageResize'),
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
    dict(
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'category_id',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'dataset_name',
            'action',
        ),
        type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_mode='topdown',
        data_root='/mnt/dataset_vol/kinect_v5_split/Val',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ground_info_mode='both_from_ground',
        pipeline=[
            dict(type='LoadImage'),
            dict(input_size=(
                256,
                256,
            ), type='EgoImageResize'),
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
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'category_id',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'dataset_name',
                    'action',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='KinectEgoposeDataset',
        use_2d_visible=True),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=None, type='CustomxRegoposeMetric', use_action=True)
val_pipeline = [
    dict(type='LoadImage'),
    dict(input_size=(
        256,
        256,
    ), type='EgoImageResize'),
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
    dict(
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'category_id',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'dataset_name',
            'action',
        ),
        type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            name='HMD_kinect_v5_flag_cascaded_ground_info_10ep',
            project='egodataset_flag_fixed_ver5-cascaded-ground-info-12dim',
            tags=[
                'kinect_v5_flag',
                'cascaded_ground_info',
                '10ep',
            ]),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                name='HMD_kinect_v5_flag_cascaded_ground_info_10ep',
                project='egodataset_flag_fixed_ver5-cascaded-ground-info-12dim',
                tags=[
                    'kinect_v5_flag',
                    'cascaded_ground_info',
                    '10ep',
                ]),
            type='WandbVisBackend'),
    ])
work_dir = 'work_dirs/HMD_kinect_v5_gbh_fh'
