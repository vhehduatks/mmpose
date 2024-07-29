#TODO metainfo 만들기 egopose
dataset_info = dict(
    dataset_name='mo2cap2',
    # paper_info=dict(
    #     author='Lin, Tsung-Yi and Maire, Michael and '
    #     'Belongie, Serge and Hays, James and '
    #     'Perona, Pietro and Ramanan, Deva and '
    #     r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
    #     title='Microsoft coco: Common objects in context',
    #     container='European conference on computer vision',
    #     year='2014',
    #     homepage='http://cocodataset.org/',
    # ),


	## [51, 153, 255] 몸통
	## [255, 128, 0] 오른쪽
	## [0, 255, 0] 왼쪽

    keypoint_info={
        0: # Neck
        dict(name='Neck', id=0, color=[51, 153, 255], type='upper', swap=''), 
        1: # LeftArm -> RightArm [51, 153, 255]
        dict( 
            name='RightArm',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='LeftArm'),
        2: # LeftForeArm -> RightForeArm [255, 128, 0] 
        dict(
            name='RightForeArm',
            id=2,
            color=[255, 128, 0] ,
            type='upper',
            swap='LeftForeArm'),
        3: # LeftHand -> RightHand [255, 128, 0] 
        dict(
            name='RightHand',
            id=3,
            color=[255, 128, 0] ,
            type='upper',
            swap='LeftHand'),
        4: # RightArm -> LeftArm [51, 153, 255]
        dict(
            name='LeftArm',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='RightArm'),
        5: # RightForeArm -> LeftForeArm [0, 255, 0]
        dict(
            name='LeftForeArm',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='RightForeArm'),
        6: # RightHand -> LeftHand [0, 255, 0]
        dict(
            name='LeftHand',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='RightHand'),
        7: # LeftUpLeg -> RightUpLeg [51, 153, 255]
        dict(
            name='RightUpLeg',
            id=7,
            color=[51, 153, 255],
            type='lower',
            swap='LeftUpLeg'),
        8: # LeftLeg -> RightLeg [255, 128, 0]
        dict(
            name='RightLeg',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='LeftLeg'),
        9: # LeftFoot -> RightFoot [255, 128, 0]
        dict(
            name='RightFoot',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='LeftFoot'),
        10: # LeftToeBase -> RightToeBase [255, 128, 0]
        dict(
            name='RightToeBase',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='LeftToeBase'),
        11: # RightUpLeg -> LeftUpLeg [51, 153, 255]
        dict(
            name='LeftUpLeg',
            id=11,
            color=[51, 153, 255],
            type='lower',
            swap='RightUpLeg'),
        12: # RightLeg -> LeftLeg [0, 255, 0]
        dict(
            name='LeftLeg',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='RightLeg'),
        13: # RightFoot -> LeftFoot [0, 255, 0]
        dict(
            name='LeftFoot',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='RightFoot'),
        14: # RightToeBase -> LeftToeBase [0, 255, 0]
        dict(
            name='LeftToeBase',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='RightToeBase'),
    },

    skeleton_info={
        0:
        dict(link=('Neck','LeftArm'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('Neck','RightArm'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('LeftArm','LeftForeArm'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('LeftForeArm','LeftHand'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('RightArm','RightForeArm'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('RightForeArm','RightHand'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('LeftArm','LeftUpLeg'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('LeftUpLeg','LeftLeg'),
            id=7,
            color=[0, 255, 0]),
        8:
        dict(link=('LeftLeg','LeftFoot'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('LeftFoot','LeftToeBase'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('RightArm','RightUpLeg'), id=10, color=[51, 153, 255]),
        11:
        dict(link=('RightUpLeg','RightLeg'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('RightLeg','RightFoot'), id=12, color=[255, 128, 0]),
        13:
        dict(link=('RightFoot','RightToeBase'), id=13, color=[255, 128, 0]),
    },
    # joint_weights=[
    #     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    # ],
    # sigmas=[
    #     0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    #     0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    # ]

	joint_weights = [1.0, 1.0, 1.2, 1.5, 1.0, 1.2, 1.5, 1.0, 1.2, 
				  1.5, 1.5, 1.0, 1.2, 1.5, 1.5],
	sigmas = [0.026, 0.079, 0.072, 0.062, 0.079, 0.072, 
		   0.062, 0.107, 0.087, 0.089, 0.089, 0.107, 0.087, 0.089, 0.089]

)
