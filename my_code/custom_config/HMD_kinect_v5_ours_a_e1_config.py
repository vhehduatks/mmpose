"""Task 21.5 E1 — minimal pesens block: global cross-attn ([z, pose, kin,
hmd] K/V) REMOVED; keeps query welding (canon_xyz + spatial_feat + PE),
self-attn, and the zero-init sensor cross-attn. Same recipe/warm-start as
pesens (cross-attn keys in the ckpt become unused; not init-identical —
simplification arm, internal-only). Prediction: ~= pesens 58.08.
"""

_base_ = ['./HMD_kinect_v5_ours_a_pesens_config.py']

model = dict(head=dict(attn_mode='replace_self'))
