"""Task 42 / Arm D1+D2 (IEEE VR 2027) — robustness sweep on the FROZEN headline model.

Test-only config: the headline recipe (HMD_kinect_v5_flag_cascaded_ground_info_10ep) with
the Val loader swapped for KinectEgoposeNoisyDataset. Sweep values are passed with
--cfg-options, e.g.
    tools/test.py my_code/custom_config/HMD_kinect_v5_t42_noise_config.py <headline.pth> \
        --cfg-options test_dataloader.dataset.sigma_ctrl_m=0.02 test_dataloader.dataset.sigma_head_m=0.02 \
        --work-dir /mnt/linux_hdd_a/mmpose_work_dirs/t42_noise/ctrl0.02
σ = 0 must reproduce the headline 64.36 (sanity gate).
"""

_base_ = ['./HMD_kinect_v5_flag_cascaded_ground_info_10ep_config.py']

custom_imports = dict(
    imports=['my_code.custom_config.ours_t42_noise_modules'],
    allow_failed_imports=False)

val_dataloader = dict(dataset=dict(
    type='KinectEgoposeNoisyDataset',
    sigma_head_m=0.0, sigma_ctrl_m=0.0, floor_offset_m=0.0, noise_seed=0))
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='CustomPose3dLocalVisualizer_xregopose',
    vis_backends=vis_backends)

work_dir = '/mnt/linux_hdd_a/mmpose_work_dirs/t42_noise'
