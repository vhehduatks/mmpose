"""Task 42 / Arm A control (IEEE VR 2027) — our single-stage image-only host, BASELINE.

Byte-identical recipe to HMD_kinect_v5_flag_cascaded_stage1_only_image_only_10ep
(the 71.17 row); re-run from the same commit as the injected sibling so the pair
is matched. Only work_dir changes (HDD).
"""

_base_ = ['./HMD_kinect_v5_flag_cascaded_stage1_only_image_only_10ep_config.py']

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends)

work_dir = '/mnt/linux_hdd_a/mmpose_work_dirs/t42_kinect_img_base'
