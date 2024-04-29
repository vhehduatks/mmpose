dataset_info = dict(
    dataset_name='egopose',
    paper_info=dict(
        author='Tome, Denis and Peluse, Patrick and Agapito, Lourdes and Badino, Hernan',
        title='xR-EgoPose: Egocentric 3D Human Pose from an HMD Camera',
        container='IEEE Transactions on Pattern Analysis and Machine Intelligence',
        year='2019',
        homepage='https://github.com/facebookresearch/xR-EgoPose',
    ),
   	keypoint_info = {
		# 0: dict(name='Hips', id=0, color=[0, 255, 0], type='upper', swap=''),
		# 1: dict(name='Spine', id=1, color=[0, 255, 0], type='upper', swap=''),
		# 2: dict(name='Spine1', id=2, color=[0, 255, 0], type='upper', swap=''),
		0: dict(name='Spine2', id=3, color=[0, 255, 0], type='upper', swap=''),
		# 4: dict(name='Neck', id=4, color=[0, 255, 0], type='upper', swap=''),
		1: dict(name='Head', id=5, color=[0, 255, 0], type='', swap=''),
		# 6: dict(name='LeftShoulder', id=6, color=[51, 153, 255], type='upper', swap='RightShoulder'),
		2: dict(name='LeftArm', id=7, color=[51, 153, 255], type='upper', swap='RightArm'),
		3: dict(name='LeftForeArm', id=8, color=[51, 153, 255], type='upper', swap='RightForeArm'),
		4: dict(name='LeftHand', id=9, color=[51, 153, 255], type='upper', swap='RightHand'),
		# 10: dict(name='RightShoulder', id=10, color=[51, 153, 255], type='upper', swap='LeftShoulder'),
		5: dict(name='RightArm', id=11, color=[51, 153, 255], type='upper', swap='LeftArm'),
		6: dict(name='RightForeArm', id=12, color=[51, 153, 255], type='upper', swap='LeftForeArm'),
		7: dict(name='RightHand', id=13, color=[255, 128, 0], type='upper', swap='LeftHand'),
		8: dict(name='LeftUpLeg', id=14, color=[255, 128, 0], type='lower', swap='RightUpLeg'),
		9: dict(name='LeftLeg', id=15, color=[255, 128, 0], type='lower', swap='RightLeg'),
		10: dict(name='LeftFoot', id=16, color=[255, 128, 0], type='lower', swap='RightFoot'),
		11: dict(name='LeftToeBase', id=17, color=[255, 128, 0], type='lower', swap='RightToeBase'),
		12: dict(name='RightUpLeg', id=18, color=[255, 128, 0], type='lower', swap='LeftUpLeg'),
		13: dict(name='RightLeg', id=19, color=[255, 128, 0], type='lower', swap='LeftLeg'),
		14: dict(name='RightFoot', id=20, color=[255, 128, 0], type='lower', swap='LeftFoot'),
		15: dict(name='RightToeBase', id=21, color=[255, 128, 0], type='lower', swap='LeftToeBase'),
	},
	skeleton_info = {
		# 0: dict(link=('Hips', 'Spine'), id=0, color=[0, 0, 0]), # p to child 
		# 1: dict(link=('Spine', 'Spine1'), id=1, color=[0, 0, 0]), # p to child 
		0: dict(link=('Spine2', 'Head'), id=2, color=[51, 153, 255]),
		# 3: dict(link=('Neck', 'LeftShoulder'), id=3, color=[51, 153, 255]),
		1: dict(link=('Spine2', 'LeftArm'), id=4, color=[51, 153, 255]),
		2: dict(link=('LeftArm', 'LeftForeArm'), id=5, color=[51, 153, 255]),
		3: dict(link=('LeftForeArm', 'LeftHand'), id=6, color=[51, 153, 255]),
		# 7: dict(link=('Neck', 'RightShoulder'), id=7, color=[51, 153, 255]),
		4: dict(link=('Spine2', 'RightArm'), id=8, color=[51, 153, 255]),
		5: dict(link=('RightArm', 'RightForeArm'), id=9, color=[0, 255, 0]),
		6: dict(link=('RightForeArm', 'RightHand'), id=10, color=[255, 128, 0]),
		7: dict(link=('Spine2', 'LeftUpLeg'), id=11, color=[255, 128, 0]),
		8: dict(link=('LeftUpLeg', 'LeftLeg'), id=12, color=[255, 128, 0]),
		9: dict(link=('LeftLeg', 'LeftFoot'), id=13, color=[0, 255, 0]),
		10: dict(link=('LeftFoot', 'LeftToeBase'), id=14, color=[0, 255, 0]),
		11: dict(link=('Spine2', 'RightUpLeg'), id=15, color=[0, 255, 0]),
		12: dict(link=('RightUpLeg', 'RightLeg'), id=16, color=[0, 255, 0]),
		13: dict(link=('RightLeg', 'RightFoot'), id=17, color=[0, 255, 0]),
		14: dict(link=('RightFoot', 'RightToeBase'), id=18, color=[0, 255, 0]),
	},
    joint_weights=[
        0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5
    ],
    sigmas=[
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.079, 0.079
    ])


# # Generate keypoint information
# keypoints = [joint["name"] for joint in new_dataset_info["joints"]]

# # Generate skeleton information
# # We need the joint IDs for start and end points to match the COCO format
# joint_name_to_id = {joint["name"]: joint["id"] for joint in new_dataset_info["joints"]}
# skeleton = [[joint_name_to_id[connection["start"]], joint_name_to_id[connection["end"]]] for connection in new_dataset_info["skeleton"]]

# # Assuming uniform joint weights and sigmas for simplicity
# # In real scenarios, these values should be determined based on the dataset characteristics
# joint_weights = [1.0] * len(keypoints)  # Uniform weight for all joints
# sigmas = [0.025] * len(keypoints)  # Uniform sigma for all joints

# keypoints, skeleton, joint_weights, sigmas