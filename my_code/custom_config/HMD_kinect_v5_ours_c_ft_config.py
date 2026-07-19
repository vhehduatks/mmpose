"""Task 15.3c — fine-tune stage 2 on the tok+s25 STF1-fused coarse.

Stage 1 + backbone frozen (FreezeStage1Hook); only stage 2 adapts to the
fused-coarse input distribution. Train fused tree from the Train-split
STF1 inference; val fused tree = the Val tree (deployment condition).
Ours-T fine-tune recipe: 5 ep, AdamW 1e-4, MultiStepLR [3,4]x0.5.
"""

_base_ = ['./HMD_kinect_v5_ours_c_frozen_config.py']

custom_imports = dict(
    imports=['my_code.custom_config.ours_c_modules',
             'my_code.custom_config.ours_t_modules'],
    allow_failed_imports=False,
)

_max_epochs = 5

train_cfg = dict(max_epochs=_max_epochs)

optim_wrapper = dict(optimizer=dict(lr=0.0001, type='AdamW'))

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=_max_epochs,
         milestones=[3, 4], gamma=0.5, by_epoch=True),
]

custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='FreezeStage1Hook'),
]

train_dataloader = dict(dataset=dict(
    fused_root='/mnt/dataset_vol/preds_coarse_tok_n25_train'))
val_dataloader = dict(dataset=dict(
    fused_root='/mnt/dataset_vol/preds_coarse_tok_n25'))
