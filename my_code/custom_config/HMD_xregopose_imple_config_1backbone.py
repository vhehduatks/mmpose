# ## mo2cap2 dataset small, test small
ann_file_test = r'F:\xr_egopose_dataset_small\TestSet'
ann_file_val = r'F:\xr_egopose_dataset_small\ValSet'
ann_file_train = r'F:\xr_egopose_dataset_small\TrainSet'
# ---
# ann_file_test = r'F:\mo2cap2_data_small\TestSet'
# ann_file_val = r'F:\mo2cap2_data_small\ValSet'
# ann_file_train = r'F:\mo2cap2_data_small\TrainSet'

auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
coco_pretrained_resnet101_256x192 = 'C:\\Users\\user\\Downloads\\pytorch-20240821T053436Z-001\\pytorch\\pose_coco\\coco_pose_resnet_101_256x192.pth.tar'
mpii_pretrained_resnet101_256x256 = 'C:\\Users\\user\\.cache\\torch\\hub\\checkpoints\\pose_resnet_101_256x256.pth.tar'
mpii_pretrained_resnet101_384x384 = 'C:\\Users\\user\\Downloads\\pose_mpii\\pose_resnet_101_384x384.pth.tar'

# coco_pretrained_resnet101_256x192 = '/workspace/mmpose/my_code/coco_pose_resnet_101_256x192.pth.tar'
# mpii_pretrained_resnet101_256x256 = '/workspace/mmpose/my_code/pose_resnet_101_256x256.pth.tar'
torchvision = 'torchvision://resnet101'


train_cfg = dict(
    # max_iters=100000, type='IterBasedTrainLoop', val_interval=1000
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1,
	)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
	optimizer=dict(lr=0.0005, type='AdamW'),
	paramwise_cfg=dict(
	custom_keys={
		# 'backbone2': dict(lr_mult=0)
	}
	)
)
param_scheduler = [
    # dict(
    #     begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        # milestones=[35000],  # Every 5000 iterations
		milestones=[i for i in range(10)],  
        gamma=0.5,
        by_epoch=True
    ),
]

default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=3,
        rule='less',
        save_best='xregopose/Full Body_All_mpjpe',
        type='CheckpointHook',
		by_epoch=True
		),
    visualization=dict(
        enable=True,
		# interval=15, 
		interval=3,
		kpt_thr=0.3, type='PoseVisualizationHook'))


randomness = dict(
	seed=42,
	diff_rank_seed=True,
    deterministic=True
	)
resume = False

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
    # sigma=[2.,2.,2.,2.,2.,2.,2.,
	# 	   3.,3.,3.,3.,3.,3.,3.,3.],
	# sigma=[3.,3.,3.,3.,3.,3.,3.,
	# 	   3.,3.,3.,3.,3.,3.,3.,3.],
	# unbiased = False,
	# blur_kernel_size= 11,
    type='Custom_mo2cap2_MSRAHeatmap')

custom_hooks = [
    dict(type='SyncBuffersHook'),
]

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
	type='TopdownPoseEstimator',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=coco_pretrained_resnet101_256x192,
            type='Pretrained'),
        type='ResNet'),
	# backbone2=dict(
    #     depth=101,
    #     init_cfg=dict(
    #         checkpoint=mpii_pretrained_resnet101_256x256,
    #         type='Pretrained'),
    #     type='ResNet'),
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
        decoder=codec,
        in_channels=2048,
        loss=dict(
			loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        loss_cosine_similarity=dict(loss_weight=1., type='cosine_similarity'), # .1
        loss_heatmap_recon=dict(
            loss_weight=500, type='KeypointMSELoss', use_target_weight=False),
        loss_limb_length=dict(loss_weight=1., type='limb_length'), # .5
        loss_pose_l2norm=dict(loss_weight=1.0, type='pose_l2norm'), # 1.
		loss_hmd = dict(type='MSELoss'),
		# loss_backbone_latant = dict(type='MSELoss',loss_weight = 1.),
		# loss_backbone_heatmap =dict(
        #     loss_weight=1000, type='KeypointMSELoss', use_target_weight=False),
        out_channels=16,
        type='CustomxRegoposeBaselinel1'),
    test_cfg=dict(
        flip_mode='heatmap',
        flip_test=False,
        output_heatmaps=False,
        shift_heatmap=False),
    )



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

data_mode = 'topdown'
dataset_type = 'CustomEgoposeDataset'
dataset_mo2cap2_train = dict(
    data_mode=data_mode,
    data_root=ann_file_train,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    # input_size=codec['input_size'],
    pipeline=train_pipeline,
    type=dataset_type)

dataset_mo2cap2_val = dict(
    data_mode=data_mode,
    data_root=ann_file_test,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    # input_size=codec['input_size'],
    pipeline=val_pipeline,
    test_mode=True,
    type=dataset_type)


torchvision = 'torchvision://resnet101'


train_dataloader = dict(
    batch_size=58,
    dataset=dataset_mo2cap2_train,
    drop_last=True,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))


val_dataloader = dict(
    batch_size=58,
    dataset=dataset_mo2cap2_val,
    drop_last=False,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=None, type='CustomxRegoposeMetric', use_action=False)

test_evaluator = dict(
    ann_file=None, type='CustomxRegoposeMetric', use_action=True)


vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(
	# 	init_kwargs=dict(project='mmpose_xregopose_baseline_recall_test'),
    #     # init_kwargs=dict(project='mmpose_mo2cap2_baseline_middle'),
	# 	# init_kwargs=dict(project='mmpose_mo2cap2_baseline_all'),
    #     type='WandbVisBackend'),
]

visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends
	)
work_dir = 'work_dirs/HMD_mo2cap2_test'
