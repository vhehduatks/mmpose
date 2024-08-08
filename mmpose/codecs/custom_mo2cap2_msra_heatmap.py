# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import (generate_gaussian_heatmaps,
									 generate_unbiased_gaussian_heatmaps)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module()
class Custom_mo2cap2_MSRAHeatmap(BaseKeypointCodec):
	"""Represent keypoints as heatmaps via "MSRA" approach. See the paper:
	`Simple Baselines for Human Pose Estimation and Tracking`_ by Xiao et al
	(2018) for details.

	Note:

		- instance number: N
		- keypoint number: K
		- keypoint dimension: D
		- image size: [w, h]
		- heatmap size: [W, H]

	Encoded:

		- heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
			where [W, H] is the `heatmap_size`
		- keypoint_weights (np.ndarray): The target weights in shape (N, K)

	Args:
		input_size (tuple): Image size in [w, h]
		heatmap_size (tuple): Heatmap size in [W, H]
		sigma (float): The sigma value of the Gaussian heatmap
		unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
			encoding. See `Dark Pose`_ for details. Defaults to ``False``
		blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
			modulation in DarkPose. The kernel size and sigma should follow
			the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
			Defaults to 11

	.. _`Simple Baselines for Human Pose Estimation and Tracking`:
		https://arxiv.org/abs/1804.06208
	.. _`Dark Pose`: https://arxiv.org/abs/1910.06278
	"""

	## 3d baseline
	# instance_mapping_table  -> gt_instance
	instance_mapping_table = dict(
		bbox='bboxes',
		bbox_score='bbox_scores',
		keypoints='keypoints',
		keypoints_cam='keypoints_cam',
		keypoints_visible='keypoints_visible',
		# In CocoMetric, the area of predicted instances will be calculated
		# using gt_instances.bbox_scales. To unsure correspondence with
		# previous version, this key is preserved here.
		bbox_scale='bbox_scales',
		# `head_size` is used for computing MpiiPCKAccuracy metric,
		# namely, PCKh
		head_size='head_size',
		keypoint3d = 'keypoint3d',
		action = 'action',
		
	)

	# items in `field_mapping_table` will be packed into
	# PoseDataSample.gt_fields and converted to Tensor. These items will be
	# used for computing losses
	field_mapping_table = dict(
		heatmaps='heatmaps',)

	# items in `label_mapping_table` will be packed into
	# PoseDataSample.gt_instance_labels and converted to Tensor. These items
	# will be used for computing losses
	# label_mapping_table -> gt_instance_labels
	label_mapping_table = dict(
		keypoint_weights='keypoint_weights',keypoint3d = 'keypoint3d',keypoints='keypoints', hmd_info = 'hmd_info', hmd_info_w_noise ='hmd_info_w_noise')


	auxiliary_encode_keys = {'keypoint3d',}
	##



	def __init__(self,
				 input_size: Tuple[int, int],
				 heatmap_size: Tuple[int, int],
				 sigma: float,
				 unbiased: bool = False,
				 blur_kernel_size: int = 11) -> None:
		super().__init__()
		self.input_size = input_size
		self.heatmap_size = heatmap_size
		self.sigma = sigma
		self.unbiased = unbiased

		# The Gaussian blur kernel size of the heatmap modulation
		# in DarkPose and the sigma value follows the expirical
		# formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
		# which gives:
		#   sigma~=3 if ks=17
		#   sigma=2 if ks=11;
		#   sigma~=1.5 if ks=7;
		#   sigma~=1 if ks=3;
		self.blur_kernel_size = blur_kernel_size
		self.scale_factor = (np.array(input_size) /
							 heatmap_size).astype(np.float32)

	def encode(self,
			keypoints: np.ndarray,
			keypoint3d: np.ndarray,
			keypoints_visible: Optional[np.ndarray] = None ) -> dict:
		"""Encode keypoints into heatmaps. Note that the original keypoint
		coordinates should be in the input image space.

		Args:
			keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
			keypoints_visible (np.ndarray): Keypoint visibilities in shape
				(N, K)

		Returns:
			dict:
			- heatmaps (np.ndarray): The generated heatmap in shape
				(K, H, W) where [W, H] is the `heatmap_size`
			- keypoint_weights (np.ndarray): The target weights in shape
				(N, K)
		"""

		assert keypoints.shape[0] == 1, (
			f'{self.__class__.__name__} only support single-instance '
			'keypoint encoding')

		if keypoints_visible is None:
			keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

		if self.unbiased:
			heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
				heatmap_size=self.heatmap_size,
				keypoints=keypoints / self.scale_factor,
				keypoints_visible=keypoints_visible,
				sigma=self.sigma)
		else:
			heatmaps, keypoint_weights = generate_gaussian_heatmaps(
				heatmap_size=self.heatmap_size,
				keypoints=keypoints / self.scale_factor,
				keypoints_visible=keypoints_visible,
				sigma=self.sigma)
		'''
		hmd 정보 임베딩으로 보내고, hmd 정보는 벡터로?..왜냐면 모델의 아웃풋이 trainset에 맞춰져 있으니까, 
		예를들어 trainset에서 xy평면에 발이 있으면, testset은 zy 평면에 발이 있는 거임.
		문제가 뭐냐, trainset은 훈련할때 gt의 좌표가 실제 pred의 gt 좌표와 일치함
		근데 testset은 gt의 좌표가 pred의 좌표와 일치하지않음.
		0. trainset의 좌표계로 testset을 수정
		1. 둘다 머리가 0,0,0으로 맞춰버릴까?.. = 카메라자체가 계속 이동함.심지어 test셋은 0,0,0이 카메라가 아님.
		2. test eval metric에 있는 rescale-prosuto 과정을 평가 때 사용하면? 
		3. hmd_info를 머리에서 팔로 가는 두개의 벡터로 ?
		4. 어차피 ego view의 좌표계는 카메라 좌표계고 그건 머리의 좌표가 000 인거랑 똑같음. 그럼 그냥 머리나 목을 000으로 치고 상대좌표화시켜버려도 된다는 소리 아닌가?
		5. 4로 가고 trainset 의 좌표계와 testset 의 좌표계를 일치시킨 다음 머리를 000으로 만들자.

		발은 근데 그냥 0에 맞춰버리면 되는데 머리랑 손은 좌표가 변하
		그래서 hmd_i
		'''



		encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights)

		return encoded

	def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Decode keypoint coordinates from heatmaps. The decoded keypoint
		coordinates are in the input image space.

		Args:
			encoded (np.ndarray): Heatmaps in shape (K, H, W)

		Returns:
			tuple:
			- keypoints (np.ndarray): Decoded keypoint coordinates in shape
				(N, K, D)
			- scores (np.ndarray): The keypoint scores in shape (N, K). It
				usually represents the confidence of the keypoint prediction
		"""
		heatmaps = encoded.copy()
		K, H, W = heatmaps.shape

		keypoints, scores = get_heatmap_maximum(heatmaps)

		# Unsqueeze the instance dimension for single-instance results
		keypoints, scores = keypoints[None], scores[None]

		if self.unbiased:
			# Alleviate biased coordinate
			keypoints = refine_keypoints_dark(
				keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

		else:
			keypoints = refine_keypoints(keypoints, heatmaps)

		# Restore the keypoint scale
		keypoints = keypoints * self.scale_factor

		return keypoints, scores
