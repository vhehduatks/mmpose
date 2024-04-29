# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from os import path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe

import re

@METRICS.register_module()
class MPJPE(BaseMetric):
	"""MPJPE evaluation metric.

	Calculate the mean per-joint position error (MPJPE) of keypoints.

	Note:
		- length of dataset: N
		- num_keypoints: K
		- number of keypoint dimensions: D (typically D = 2)

	Args:
		mode (str): Method to align the prediction with the
			ground truth. Supported options are:

				- ``'mpjpe'``: no alignment will be applied
				- ``'p-mpjpe'``: align in the least-square sense in scale
				- ``'n-mpjpe'``: align in the least-square sense in
					scale, rotation, and translation.

		collect_device (str): Device name used for collecting results from
			different ranks during distributed training. Must be ``'cpu'`` or
			``'gpu'``. Default: ``'cpu'``.
		prefix (str, optional): The prefix that will be added in the metric
			names to disambiguate homonymous metrics of different evaluators.
			If prefix is not provided in the argument, ``self.default_prefix``
			will be used instead. Default: ``None``.
		skip_list (list, optional): The list of subject and action combinations
			to be skipped. Default: [].
	"""

	ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale'}
	
	_NAMES = [
		'Gesticuling', 'Reacting', 'Greeting',
		'Talking', 'UpperStretching', 'Gaming',
		'LowerStretching', 'Patting', 'Walking', 'All'
	]

	_ACTION = {
		'anim_Clip1': 8, 'Opening_A_Lid': 0, 'Dribble': 5, 'Boxing': 5,
		'Standing_Arguing__1_': 3, 'Happy': 3, 'Plotting': 3, 'Counting': 4,
		'Standing_Arguing': 0, 'Standing_2H_Cast_Spell_01': 4, 'Shooting_Gun': 5,
		'Two_Hand_Spell_Casting': 0, 'Shaking_Hands_2': 2, 'Hands_Forward_Gesture': 2,
		'Rifle_Punch': 1, 'Baseball_Umpire': 5, 'Angry_Gesture': 0, 'Waving_Gesture': 0,
		'Taunt_Gesture': 0, 'Golf_Putt_Failure': 5, 'Rejected': 1, 'Shake_Fist': 2,
		'Revealing_Dice': 5, 'Golf_Putt_Failure__1_': 5, 'No': 3, 'Angry_Point': 1,
		'Agreeing': 3, 'Sitting_Thumbs_Up': 6, 'Standing_Thumbs_Up': 4, 'Patting': 7,
		'Petting': 7, 'Petting_Animal': 7, 'Taking_Punch': 0,
		'Standing_1H_Magic_Attack_01': 4, 'Talking': 3, 'Standing_Greeting': 2,
		'Happy_Hand_Gesture': 0, 'Dismissing_Gesture': 1, 'Strong_Gesture': 1,
		'Pointing_Gesture': 1, 'Golf_Putt_Victory': 5, 'Pointing': 0,
		'Thinking': 4, 'Loser': 1, 'Reaching_Out': 3, 'Crazy_Gesture': 0,
		'Golf_Putt_Victory__1_': 5, 'Insult': 3, 'Arm_Gesture': 0,
		'Beckoning': 1, 'Charge': 5, 'Weight_Shift_Gesture': 8,
		'Pain_Gesture': 1, 'Fist_Pump': 0, 'Terrified': 1, 'Surprised': 1,
		'Clapping': 1, 'Rallying': 1, 'Hand_Raising': 0, 'Sitting_Disapproval': 6,
		'Quick_Formal_Bow': 2, 'Counting__1_': 0, 'Tpose_Take_001': 4,
		'upper_stretching': 4, 'lower_stretching': 6, 'walking': 8
	}

	def __init__(self,
				 mode: str = 'mpjpe',
				 collect_device: str = 'cpu',
				 prefix: Optional[str] = None,
				 skip_list: List[str] = []) -> None:
		super().__init__(collect_device=collect_device, prefix=prefix)
		allowed_modes = self.ALIGNMENT.keys()
		if mode not in allowed_modes:
			raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
						   f"'n-mpjpe', but got '{mode}'.")

		self.mode = mode
		self.skip_list = skip_list
		self.action_map = {}
		for k, v in self._ACTION.items():
			self.action_map[k] = self._NAMES[v]


	def _map_action_name(self, name):
		additional = re.findall(r'_mixamo_com.*', name)
		if additional:
			name = name.replace(additional[0], '')
		if name in list(self.action_map.keys()):
			return self.action_map[name]
		return 'All'


	def process(self, data_batch: Sequence[dict],
				data_samples: Sequence[dict]) -> None:
		"""Process one batch of data samples and predictions. The processed
		results should be stored in ``self.results``, which will be used to
		compute the metrics when all batches have been processed.

		Args:
			data_batch (Sequence[dict]): A batch of data
				from the dataloader.
			data_samples (Sequence[dict]): A batch of outputs from
				the model.
		"""
		for data_sample in data_samples:

			pred_coords = data_sample['pred_instances']['keypoints']
			if pred_coords.ndim == 4:
				pred_coords = np.squeeze(pred_coords, axis=0)
			# ground truth data_info
			gt = data_sample['gt_instances']
			gt_coords = gt['lifting_target']

			
			#TODO 마스크 변경, json에서 action 가져와서 변경 
			# mask = gt['lifting_target_visible'].astype(bool).reshape(
			#     gt_coords.shape[0], -1)
			# # instance action
			# img_path = data_sample['target_img_path'][0]
			# _, rest = osp.basename(img_path).split('_', 1)
			# action, _ = rest.split('.', 1)
			# actions = np.array([action] * gt_coords.shape[0])

			# subj_act = osp.basename(img_path).split('.')[0]
			# if subj_act in self.skip_list:
			#     continue
			action = np.array(data_sample['action'])
			# default mask to all ones
			mask = np.ones((1, pred_coords.shape[0]), dtype=bool)
		

			result = {
				'pred_coords': pred_coords,
				'gt_coords': gt_coords,
				'mask': mask,
				'action': action
			}

			self.results.append(result)

	def compute_metrics(self, results: list) -> Dict[str, float]:
		"""Compute the metrics from processed results.

		Args:
			results (list): The processed results of each batch.

		Returns:
			Dict[str, float]: The computed metrics. The keys are the names of
			the metrics, and the values are the corresponding results.
		"""
		logger: MMLogger = MMLogger.get_current_instance()

		# pred_coords: [N, K, D]
		pred_coords = np.concatenate(
			[result['pred_coords'] for result in results])
		# gt_coords: [N, K, D]
		gt_coords = np.concatenate([result['gt_coords'] for result in results])
		# mask: [N, K]
		mask = np.concatenate([result['mask'] for result in results])
		# action_category_indices: Dict[List[int]]
		action_category_indices = defaultdict(list)
		actions = np.concatenate([result['action'] for result in results])
		for idx, action in enumerate(actions):
			# action_category = action.split('_')[0]
			action_category = self._map_action_name(action)
			action_category_indices[action_category].append(idx)

		error_name = self.mode.upper()

		logger.info(f'Evaluating {self.mode.upper()}...')
		metrics = dict()

		metrics[error_name] = keypoint_mpjpe(pred_coords, gt_coords, alignment = self.ALIGNMENT[self.mode])

		for action_category, indices in action_category_indices.items():
			metrics[f'{error_name}_{action_category}'] = keypoint_mpjpe(
				pred_coords[indices], gt_coords[indices], alignment = self.ALIGNMENT[self.mode])

		return metrics
