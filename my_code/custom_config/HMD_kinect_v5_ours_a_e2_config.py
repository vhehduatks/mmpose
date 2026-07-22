"""Task 21.5 E2 — ego-cam sensor consultation: EXACTLY the Task-18 nocanon
config (raw ego-cam joint tokens, 18a gated residual, no PE; existing
63.83 internal reference) + zero-init sensor cross-attn over 2 ego-cam
controller tokens (inv(cam2world) @ ctrl_world). The hmd token is dropped
BY CONSTRUCTION (in ego-cam it is the rigid-mount constant — zero
information). Isolates whether the controller gain needs the CANONICAL
frame or only a SHARED metric frame.
"""

_base_ = ['./HMD_kinect_v5_ours_a_nocanon_config.py']

codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='OursSensorCodec'
)

_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
)
_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

model = dict(head=dict(decoder=codec, sensor_mode='floor',
                       sensor_frame='egocam', sens_subset='ctrl'))

train_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                     pipeline=_pipeline))
val_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                   pipeline=_pipeline))
