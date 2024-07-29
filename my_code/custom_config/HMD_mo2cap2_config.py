_base_ = r'C:\Users\user\Documents\GitHub\mmpose\configs\_base_\default_runtime.py'

# runtime
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1,
    dynamic_intervals=[(280, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
	visualization=dict(type='PoseVisualizationHook', enable=True, interval = 15,kpt_thr=0.3),
	)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
# model
codec = dict(
    type='Custom_mo2cap2_MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        type='CustomMo2Cap2HeatmapHead',
        in_channels=2048,
        out_channels=15, # keypoint num
        loss=dict(type='KeypointMSELoss', use_target_weight=True, loss_weight = 1000),
		loss_3d=dict(type='MSELoss', use_target_weight=False, loss_weight = 10),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
		output_heatmaps=True,
    ))

# data
dataset_type = 'Mo2Cap2CocoDataset'
data_mode = 'topdown'
data_root = r'C:\Users\user\Documents\GitHub\mmpose\data'

# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_72'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_73' # 68:1000개 짜리 ,64 : 10개짜리, 66 : 500개짜리, 67 : 50개 짜리

# dataset config
# ---
# temp val:10,train:50
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_64'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_67'
# ---
# fov : 60
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_70'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_71'
# ---
# fov : 100
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_72'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_73'
# ---
# # fov : 150
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_74'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_75'
# ---
# # fov : 60, in the room
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_79'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_78'
# ---
# fov : 100, in the room
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_80'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_81'
# ---
# # fov : 150, in the room
# ann_file_val = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_83'
# ann_file_train = r'C:\Users\user\AppData\LocalLow\DefaultCompany\perception tutorial\solo_82'
# ---
# # mo2cap2 dataset temp
# ann_file_val = r'F:\mo2cap2_data_temp_extracted\TestSet'
# ann_file_train = r'F:\mo2cap2_data_temp_extracted\TrainSet'
# ---
# ## mo2cap2 dataset small, test small
# ann_file_test = r'F:\mo2cap2_data_small\TestSet'
# ann_file_val = r'F:\mo2cap2_data_small\ValSet'
# ann_file_train = r'F:\mo2cap2_data_small\TrainSet'
# ---
# mo2cap2 dataset train middel, test all
ann_file_test = r'F:\mo2cap2_data_half\TestSet'
ann_file_val = r'F:\mo2cap2_data_half\ValSet'
ann_file_train = r'F:\mo2cap2_data_half\TrainSet'
# ---
# # mo2cap2 dataset train all, test all
# ann_file_val = r'F:\extracted_mo2cap2_dataset\TestSet'
# ann_file_train = r'F:\extracted_mo2cap2_dataset\TrainSet'


train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.),
    dict(type='RandomFlip', direction='horizontal'),
    # # dict(type='RandomHalfBody'),
    # # dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.),
    dict(type='TopdownAffine', input_size=codec['input_size']),
	dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.),
    dict(type='TopdownAffine', input_size=codec['input_size']),
	# dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

dataset_mo2cap2_train = dict(
	type=dataset_type,
    # data_root=data_root,
    data_mode=data_mode,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file=ann_file_train,
    # data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
    pipeline=train_pipeline,
)

dataset_mo2cap2_val = dict(
	type=dataset_type,
    # data_root=data_root,
    data_mode=data_mode,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file=ann_file_val,
    # data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
    pipeline=val_pipeline,
	test_mode = False,
)

dataset_mo2cap2_test = dict(
	type=dataset_type,
    # data_root=data_root,
    data_mode=data_mode,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file=ann_file_test,
    # data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
    pipeline=test_pipeline,
	test_mode = True,
)

# 1000,500 할때는 16:2,8:2
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
	drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_mo2cap2_train)


val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dataset_mo2cap2_val
	)

## test_data : no 2d kpts
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dataset_mo2cap2_test
	)

# evaluators
# evaluators
# 추가할 수 있으면 Custom_MPJPE
val_evaluator = dict(
    type='CustomMo2Cap2Metric',
	ann_file=None,
	use_action = False,
	)
test_evaluator = dict(
    type='CustomMo2Cap2Metric',
	ann_file=None,
	use_action = True,
	)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    dict(
		type='WandbVisBackend',
		init_kwargs=dict(
			# entity = "cv04",
			project="mmpose_mo2cap2dataset",
			),
		),
]
visualizer = dict(
    # type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer'
	# type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer'
	type = 'CustomPose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer'
	)

# custom_hooks = [
#     dict(
#         type='YOLOXPoseModeSwitchHook',
#         num_last_epochs=20,
#         new_train_pipeline=train_pipeline_stage2,
#         priority=48),
#     dict(type='SyncNormHook', priority=48),
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0002,
#         update_buffers=True,
#         strict_load=False,
#         priority=49),
# ]
