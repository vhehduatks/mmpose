"""Task 21.2 CONTROL — baked-in-conditioning: the EXISTING relative hmd_info
parametrization re-tokenized as 3 K/V tokens [right_local, left_local,
GBH heights] (distances dropped: redundant norms of the locals). Same
params/token count as sensC; relative representation instead of floor
xyz. If sensC ties this, the representation claim dies.
"""

_base_ = ['./HMD_kinect_v5_ours_a_t01_config.py']

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

model = dict(head=dict(decoder=codec, sensor_mode='baked'))

train_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                     pipeline=_pipeline))
val_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                   pipeline=_pipeline))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_t01/best_xregopose_Full Body_All_mpjpe_epoch_2.pth'
