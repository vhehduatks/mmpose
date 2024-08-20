_base_ = r'C:\Users\user\Documents\GitHub\mmpose\configs\_base_\default_runtime.py'

# runtime
# train_cfg = dict(
# 	_delete_=True,
# 	type='EpochBasedTrainLoop',
# 	max_epochs=10,
# 	val_interval=1,
# 	dynamic_intervals=[(280, 1)])
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',  # Change to iteration-based training loop
    max_iters=500,            # Set the maximum number of iterations
    val_interval=50           # Validation interval in iterations
)
auto_scale_lr = dict(base_batch_size=256)

# default_hooks = dict(
# 	checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
# 	visualization=dict(type='PoseVisualizationHook', enable=True, interval = 15,kpt_thr=0.3),
# 	)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10000, max_keep_ckpts=3),
    visualization=dict(type='PoseVisualizationHook', enable=True, interval=15, kpt_thr=0.3),
)

# Early stopping hook configuration
early_stopping_hook = dict(
    type='EarlyStoppingHook',
    monitor='mo2cap2/Full Body_All_mpjpe',  # Metric to monitor
    rule='less',  # 'less' since MPJPE is an error metric
    min_delta=1,  # Minimum change to qualify as improvement
    patience=3,  # Number of validation intervals to wait
    strict=False,  # Do not crash if the metric is not found
    check_finite=True,  # Stop if the metric becomes NaN or infinite
    stopping_threshold=None  # No immediate stopping threshold
)

# Add the early stopping hook to the list of custom hooks
custom_hooks = [
    early_stopping_hook,
    # Add other hooks if necessary
]



# # optimizer
# optim_wrapper = dict(optimizer=dict(
# 	type='AdamW',
# 	lr=5e-4,
# ))

# # learning policy
# param_scheduler = [
# 	dict(
# 		type='LinearLR', begin=0, end=500, start_factor=0.001,
# 		by_epoch=False),  # warm-up
# 	dict(
# 		type='MultiStepLR',
# 		begin=0,
# 		end=210,
# 		milestones=[170, 200],
# 		gamma=0.1,
# 		by_epoch=True)
# ]

param_scheduler = [
    dict(
        type='LinearLR', 
        begin=0, 
        end=500, 
        start_factor=0.001,
        by_epoch=False
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=70000,
        milestones=[5000 * i for i in range(1, 15)],  # Every 5000 iterations
        gamma=0.5,
        by_epoch=False
    )
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
    )
)

# # learning policy
# param_scheduler = [
#     dict(
#         type='ReduceOnPlateauLR',
#         mode='min',
#         factor=0.1,
#         patience=7,  # Adjusted to match your es_patience-3
#         threshold=0.0001,
#         threshold_mode='rel',
#         cooldown=0,
#         min_lr=1e-8,
#         eps=1e-08,
#         verbose=True
#     )
# ]

# model
codec = dict(
	type='Custom_mo2cap2_MSRAHeatmap', input_size=(256, 256), heatmap_size=(47, 47), sigma=3)
#mpii pretrained path 
mpii_pretrained_resnet101_256x256 = r'C:\Users\user\.cache\torch\hub\checkpoints\pose_resnet_101_256x256.pth.tar'
mpii_pretrained_resnet101_384x384 = r'C:\Users\user\.cache\torch\hub\checkpoints\pose_resnet_101_384x384.pth.tar'
#torchvision pretrained path
torchvision = 'torchvision://resnet101'

model = dict(
	type='TopdownPoseEstimator',
	data_preprocessor=dict(
		type='PoseDataPreprocessor',
		mean=[123.675, 116.28, 103.53],
		std=[58.395, 57.12, 57.375],
		bgr_to_rgb=True),
	backbone=dict(
		type='ResNet',
		depth=101,
		init_cfg=dict(type='Pretrained', checkpoint=mpii_pretrained_resnet101_256x256),
	),
	head=dict(
		type='CustomMo2Cap2Baselinel1',
		in_channels=2048,
		out_channels=15, # keypoint num
		loss=dict(type='KeypointMSELoss', use_target_weight=True, loss_weight = 1000),
		loss_pose_l2norm = dict(type='pose_l2norm', loss_weight = 1.),
		loss_cosine_similarity = dict(type='cosine_similarity', loss_weight = 0.1),
		loss_limb_length = dict(type='limb_length', loss_weight = 0.5),
		loss_heatmap_recon = dict(type='KeypointMSELoss', use_target_weight=True, loss_weight = 500),
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
# data_root = r'C:\Users\user\Documents\GitHub\mmpose\data'

# ---
# # mo2cap2 dataset temp
# ann_file_val = r'F:\mo2cap2_data_temp_extracted\TestSet'
# ann_file_train = r'F:\mo2cap2_data_temp_extracted\TrainSet'
# ---
# # ## mo2cap2 dataset small, test small
ann_file_test = r'F:\mo2cap2_data_small\TestSet'
ann_file_val = r'F:\mo2cap2_data_small\ValSet'
ann_file_train = r'F:\mo2cap2_data_small\TrainSet'
# ---
# mo2cap2 dataset train middel, test all
# ann_file_test = r'F:\mo2cap2_data_half\TestSet'
# ann_file_val = r'F:\mo2cap2_data_half\ValSet'
# ann_file_train = r'F:\mo2cap2_data_half\TrainSet'
# ---
# # mo2cap2 dataset train all, test all
# ann_file_val = r'F:\extracted_mo2cap2_dataset\TestSet'
# ann_file_train = r'F:\extracted_mo2cap2_dataset\TrainSet'

####
# ann_file_test = r'F:\mo2cap2_one_data\TestSet'
# ann_file_val = r'F:\mo2cap2_one_data\ValSet'
# ann_file_train = r'F:\mo2cap2_one_data\TrainSet'
###

# ann_file_test = r'D:\cross_plat\mo2cap2_data_half\TestSet'
# ann_file_val = r'D:\cross_plat\mo2cap2_data_half\ValSet'
# ann_file_train = r'D:\cross_plat\mo2cap2_data_half\TrainSet'


train_pipeline = [
	dict(type='LoadImage'),
	dict(type='GetBBoxCenterScale',padding=1.),
	# dict(type='RandomFlip', direction='horizontal'), # 3d keypoints도 플립시켜야됨.
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
# test_pipeline = [
#     dict(type='LoadImage'),
#     dict(type='GetBBoxCenterScale',padding=1.),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
# 	# dict(type='GenerateTarget', encoder=codec),
#     dict(type='PackPoseInputs')
# ]

dataset_mo2cap2_train = dict(
	type=dataset_type,
	data_root=ann_file_train,
	data_mode=data_mode,
	filter_cfg=dict(filter_empty_gt=False, min_size=32),
	# ann_file=ann_file_train,
	# data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
	pipeline=train_pipeline,
	input_size=codec['input_size'],
)

dataset_mo2cap2_val = dict(
	type=dataset_type,
	data_root=ann_file_test,
	data_mode=data_mode,
	filter_cfg=dict(filter_empty_gt=False, min_size=32),
	# ann_file=ann_file_val,
	# data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
	pipeline=val_pipeline,
	test_mode = True,
	input_size=codec['input_size'],
)

# dataset_mo2cap2_test = dict(
# 	type=dataset_type,
#     data_root=ann_file_test,
#     data_mode=data_mode,
#     filter_cfg=dict(filter_empty_gt=False, min_size=32),
#     # ann_file=ann_file_test,
#     # data_prefix=dict(img=r'C:\Users\user\Documents\GitHub\mmpose\data\coco\train2017'),
#     pipeline=test_pipeline,
# 	test_mode = True,
# )

# 1000,500 할때는 16:2,8:2
train_dataloader = dict(
	batch_size=64,
	num_workers=6,
	persistent_workers=False,
	pin_memory=True,
	drop_last=True,
	sampler=dict(type='DefaultSampler', shuffle=True, round_up=True),
	dataset=dataset_mo2cap2_train)


val_dataloader = dict(
	batch_size=64,
	num_workers=6,
	persistent_workers=False,
	pin_memory=True,
	drop_last=False,
	sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
	dataset=dataset_mo2cap2_val
	)

test_dataloader =val_dataloader
# ## test_data : no 2d kpts
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=0,
#     persistent_workers=False,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
#     dataset=dataset_mo2cap2_test
# 	)

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
	# # dict(type='TensorboardVisBackend'),
	dict(
		type='WandbVisBackend',
		init_kwargs=dict(
			# entity = "cv04",
			project="mmpose_mo2cap2_baseline_earlystop_test",
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
