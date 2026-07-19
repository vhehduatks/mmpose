"""Task 15.3b step 2 — frozen stage 2 fed the tokens+s25 fused coarse."""

_base_ = ['./HMD_kinect_v5_ours_c_frozen_config.py']

train_dataloader = dict(dataset=dict(fused_root='/mnt/dataset_vol/preds_coarse_tok_n25'))
val_dataloader = dict(dataset=dict(fused_root='/mnt/dataset_vol/preds_coarse_tok_n25'))
