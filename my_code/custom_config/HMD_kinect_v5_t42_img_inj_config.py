"""Task 42 / Arm A control (IEEE VR 2027) — our single-stage image-only host + FROZEN hmd12 injection.

INJECTED member of the pair with HMD_kinect_v5_t42_img_base_config.py. The
dataset now emits the REAL 12-dim hmd12 (ground_info_mode='both_from_ground',
use_hmd=True); OursSplitHMD12 routes it to the adapter and zeroes the head's
own HMD input (9 zeros, as in the baseline); head = OursInjectCascadedEnhanced.
"""

_base_ = ['./HMD_kinect_v5_t42_img_base_config.py']

custom_imports = dict(
    imports=['my_code.custom_config.ours_t42_inject_modules'],
    allow_failed_imports=False)

codec = dict(
    heatmap_size=(47, 47),
    input_size=(256, 256),
    sigma=3,
    type='Custom_mo2cap2_MSRAHeatmap'
)

model = dict(
    head=dict(
        type='OursInjectCascadedEnhanced',
        inject_stats='quest',
    ),
)

_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='OursSplitHMD12', hmd_dim=9),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='EgoImageResize', input_size=(256, 256)),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='OursSplitHMD12', hmd_dim=9),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

train_dataloader = dict(dataset=dict(
    ground_info_mode='both_from_ground', use_hmd=True, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(
    ground_info_mode='both_from_ground', use_hmd=True, pipeline=val_pipeline))
test_dataloader = val_dataloader

work_dir = '/mnt/linux_hdd_a/mmpose_work_dirs/t42_kinect_img_inj'
