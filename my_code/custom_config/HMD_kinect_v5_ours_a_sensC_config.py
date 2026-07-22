"""Task 21.2 arm C — floor-frame sensor cross-attn tokens ADDED to the t01
model (stage-1 fusion stays ON). 3 K/V tokens [ctrl_L, ctrl_R, hmd] in the
SAME floor frame as the canonicalized joints; separate zero-init sensor
cross-attn => init == t01 digit-for-digit. Derived hmd_info mod-token
dropped (was inert in Task 18). Judge vs sensbaked + sensconst controls.
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

# mod_hmd KEPT (inert per Task 18) so the zero-init sensor path preserves
# digit-for-digit warm-start identity from t01 (dropping a K/V token would
# change the cross-attn softmax at init). Arm B drops it instead.
model = dict(head=dict(decoder=codec, sensor_mode='floor'))

train_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                     pipeline=_pipeline))
val_dataloader = dict(dataset=dict(type='KinectEgoposeSensorDataset',
                                   pipeline=_pipeline))

load_from = '/home/hyeonghwan/github/mmpose/work_dirs/ours_a_t01/best_xregopose_Full Body_All_mpjpe_epoch_2.pth'
