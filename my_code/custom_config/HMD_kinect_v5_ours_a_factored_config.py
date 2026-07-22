"""Task 21.3 — factored query/evidence block (the user's intended
topology): pure-geometry query (canon_xyz + joint PE) -> self-attn ->
cross-attn over 22 keys (16 per-joint spatial tokens w/ PE + z/pose/kin
globals + 3 floor-frame sensors). Base = full pesens configuration;
backbone + stage 1 FROZEN as always; retrained 18b-style from the
headline ckpt (no warm-start guarantee possible — query embed 67->3).
Comparator: pesens internal 58.08 (the LARGER-capacity mixed-topology
incumbent — see params note in RESULTS).
"""

_base_ = ['./HMD_kinect_v5_ours_a_sensC_config.py']

model = dict(head=dict(attn_mode='factored', pe_mode='fixed'))

load_from = '/mnt/dataset_vol/work_dir_260408/HMD_kinect_v5_flag_cascaded_ground_info_10ep/best_xregopose_Full Body_All_mpjpe_epoch_10.pth'
