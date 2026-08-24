"""Task 42 / Arm A2 (IEEE VR 2027) — xR-EgoPose (Tome) + FROZEN hmd12 injection.

INJECTED member of the matched pair with HMD_xregopose_t42_tome_base_config.py.
Identical recipe; the only differences: (1) EnhanceHMDInfo(both_from_ground)
builds the 12-dim pseudo-hmd12 from GT joints (ORACLE — label it wherever it
appears), (2) OursSplitHMD12 routes it to the adapter and zeroes the head's own
HMD input, (3) head = OursInjectBaselinel1 (Tome head + adapter on feats[-1]).
"""

_base_ = ['./HMD_xregopose_t42_tome_base_config.py']

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
        type='OursInjectBaselinel1',
        inject_stats='xr',
    ),
)

_meta_keys = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'action',
    'h5_cache_path', 'h5_img_idx'
)

train_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo', mode='both_from_ground'),
    dict(type='OursSplitHMD12', hmd_dim=9),
    dict(type='PackPoseInputs', meta_keys=_meta_keys),
]

val_pipeline = [
    dict(type='LoadImageFromH5Cache'),
    dict(padding=1.0, type='GetBBoxCenterScale'),
    dict(input_size=(256, 256), type='TopdownAffine'),
    dict(encoder=codec, type='GenerateTarget'),
    dict(type='EnhanceHMDInfo', mode='both_from_ground'),
    dict(type='OursSplitHMD12', hmd_dim=9),
    dict(type='PackPoseInputs', meta_keys=_meta_keys, pack_transformed=True),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

work_dir = '/mnt/linux_hdd_a/mmpose_work_dirs/t42_xr_tome_inj'
