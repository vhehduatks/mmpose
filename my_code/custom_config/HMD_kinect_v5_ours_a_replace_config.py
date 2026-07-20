"""Task 18b — full replacement: the decoder block IS stage 2 (no MLP,
no replicated globals). Retrained with the headline 10-ep recipe from the
headline checkpoint (stage 1 + backbone frozen); no warm-start guarantee.
"""

_base_ = ['./HMD_kinect_v5_ours_a_config.py']

model = dict(head=dict(attn_mode='replace'))

_max_epochs = 10

train_cfg = dict(max_epochs=_max_epochs)

optim_wrapper = dict(optimizer=dict(lr=0.0005, type='AdamW'))

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=_max_epochs,
         milestones=[5, 8], gamma=0.5, by_epoch=True),
]
