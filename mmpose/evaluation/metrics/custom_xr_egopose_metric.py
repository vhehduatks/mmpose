# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import re

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MessageHub, MMLogger, print_log
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from mmpose.structures.bbox import bbox_xyxy2xywh
from ..functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
						  transform_sigmas)

import torch
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
								 OptSampleList, Predictions, InstanceList)

from . import mo2cap2_evaluate

@METRICS.register_module()
class CustomxRegoposeMetric(BaseMetric):
	default_prefix: Optional[str] = 'xregopose'

	def __init__(self,
				ann_file: Optional[str] = None,
				use_action: bool = False,
				use_area: bool = True,
				iou_type: str = 'keypoints',
				score_mode: str = 'bbox_keypoint',
				keypoint_score_thr: float = 0.2,
				nms_mode: str = 'oks_nms',
				nms_thr: float = 0.9,
				format_only: bool = False,
				pred_converter: Dict = None,
				gt_converter: Dict = None,
				outfile_prefix: Optional[str] = None,
				collect_device: str = 'cpu',
				prefix: Optional[str] = None,
				 ) -> None:
		super().__init__(collect_device=collect_device, prefix=prefix)
		self.ann_file = ann_file
		# initialize coco helper with the annotation json file
		# if ann_file is not specified, initialize with the converted dataset
		if ann_file is not None:
			with get_local_path(ann_file) as local_path:
				self.coco = COCO(local_path)
		else:
			self.coco = None

		self.use_area = use_area
		self.iou_type = iou_type

		allowed_score_modes = ['bbox', 'bbox_keypoint', 'bbox_rle', 'keypoint']
		if score_mode not in allowed_score_modes:
			raise ValueError(
				"`score_mode` should be one of 'bbox', 'bbox_keypoint', "
				f"'bbox_rle', but got {score_mode}")
		self.score_mode = score_mode
		self.keypoint_score_thr = keypoint_score_thr

		allowed_nms_modes = ['oks_nms', 'soft_oks_nms', 'none']
		if nms_mode not in allowed_nms_modes:
			raise ValueError(
				"`nms_mode` should be one of 'oks_nms', 'soft_oks_nms', "
				f"'none', but got {nms_mode}")
		self.nms_mode = nms_mode
		self.nms_thr = nms_thr

		if format_only:
			assert outfile_prefix is not None, '`outfile_prefix` can not be '\
				'None when `format_only` is True, otherwise the result file '\
				'will be saved to a temp directory which will be cleaned up '\
				'in the end.'
		elif ann_file is not None:
			# do evaluation only if the ground truth annotations exist
			assert 'annotations' in load(ann_file), \
				'Ground truth annotations are required for evaluation '\
				'when `format_only` is False.'

		self.format_only = format_only
		self.outfile_prefix = outfile_prefix
		self.pred_converter = pred_converter
		self.gt_converter = gt_converter


		## mo2cap2 baseline
		self.use_action = use_action

		##


	@property
	def dataset_meta(self) -> Optional[dict]:
		"""Optional[dict]: Meta info of the dataset."""
		return self._dataset_meta

	@dataset_meta.setter
	def dataset_meta(self, dataset_meta: dict) -> None:
		"""Set the dataset meta info to the metric."""
		if self.gt_converter is not None:
			dataset_meta['sigmas'] = transform_sigmas(
				dataset_meta['sigmas'], self.gt_converter['num_keypoints'],
				self.gt_converter['mapping'])
			dataset_meta['num_keypoints'] = len(dataset_meta['sigmas'])
		self._dataset_meta = dataset_meta

		if self.coco is None:
			pass


	def process(self, data_batch: Sequence[dict],
				data_samples: Sequence[dict]) -> None:

		for data_sample in data_samples:
			if 'pred_instances' not in data_sample:
				raise ValueError(
					'`pred_instances` are required to process the '
					f'predictions results in {self.__class__.__name__}. ')

			# keypoints.shape: [N, K, 2],
			# N: number of instances, K: number of keypoints
			# for topdown-style output, N is usually 1, while for
			# bottomup-style output, N is the number of instances in the image
			keypoints = data_sample['pred_instances']['keypoints']
			# [N, K], the scores for all keypoints of all instances
			keypoint_scores = data_sample['pred_instances']['keypoint_scores']
			assert keypoint_scores.shape == keypoints.shape[:2]

			# parse prediction results
			pred = dict()
			# pred['id'] = data_sample['id']
			# pred['id'] = data_sample['img_id']
			# pred['img_id'] = data_sample['img_id']

			pred['keypoints'] = keypoints
			pred['keypoint_scores'] = keypoint_scores
			# pred['category_id'] = data_sample.get('category_id', 1)

			## 3d baseline
			pred['keypoint3d'] = data_sample['pred_instances']['keypoint_3d']
			# Shape validation
			assert pred['keypoint3d'].shape[-2:] == (16, 3), \
				f"Expected pred keypoint3d shape [..., 16, 3], got {pred['keypoint3d'].shape}"
			##


			# parse gt
			gt = dict()
			if self.coco is None:
				gt['width'] = data_sample['ori_shape'][1]
				gt['height'] = data_sample['ori_shape'][0]
				# gt['img_id'] = data_sample['img_id']
				# if self.iou_type == 'keypoints_crowd':
				# 	assert 'crowd_index' in data_sample, \
				# 		'`crowd_index` is required when `self.iou_type` is ' \
				# 		'`keypoints_crowd`'
				# 	gt['crowd_index'] = data_sample['crowd_index']
				# assert 'raw_ann_info' in data_sample, \
				# 	'The row ground truth annotations are required for ' \
				# 	'evaluation when `ann_file` is not provided'
				# anns = data_sample['raw_ann_info']
				# gt['raw_ann_info'] = anns if isinstance(anns, list) else [anns]
			## 3d baseline
			gt['keypoint3d'] = data_sample['gt_instance_labels']['keypoint3d']
			# Shape validation
			assert gt['keypoint3d'].shape[-2:] == (16, 3), \
				f"Expected gt keypoint3d shape [..., 16, 3], got {gt['keypoint3d'].shape}"
			##

			## mo2cap2
			if self.use_action:
				gt['action'] = data_sample['gt_instances']['action'][0]
			##

## TODO metric 수정할 것 gt , pred 둘다 data_sample에 있ㅇ므
			# add converted result to the results list
			self.results.append((pred, gt))



	def compute_metrics(self, results: list) -> Dict[str, float]:

		logger: MMLogger = MMLogger.get_current_instance()

		# split prediction and gt list
		preds, gts = zip(*results)

		pred_list = []
		gt_list = []
		batch_actions = []

		for pred_, gt_ in zip(preds, gts):
			kpt3d = pred_['keypoint3d']
			if isinstance(kpt3d, np.ndarray):
				kpt3d = torch.from_numpy(kpt3d)
			pred_list.append(kpt3d)

			gt_kpt3d = gt_['keypoint3d']
			if isinstance(gt_kpt3d, np.ndarray):
				gt_kpt3d = torch.from_numpy(gt_kpt3d)
			gt_list.append(gt_kpt3d)

			if self.use_action:
				batch_actions.append(gt_['action'])

		# squeeze(dim=1)로 instance 차원만 제거 (N=1일 때 batch 차원 보존)
		pred_all = torch.stack(pred_list).squeeze(dim=1)  # (N, 16, 3)
		gt_all = torch.stack(gt_list).squeeze(dim=1)      # (N, 16, 3)

		# ===== Vectorized MPJPE (baseline mode) =====
		# Per-joint L2 error: (N, 16)
		per_joint_error = torch.sqrt(
			((pred_all - gt_all) ** 2).sum(dim=-1)
		) * 1000.0
		per_joint_error_np = per_joint_error.numpy()

		# Joint indices (baseline mode)
		UPPER = [0, 1, 2, 3, 4, 5, 6, 7]
		LOWER = [8, 9, 10, 11, 12, 13, 14, 15]

		# Per-sample MPJPE for each body part: (N,)
		full_body_errors = per_joint_error_np.mean(axis=1)
		upper_body_errors = per_joint_error_np[:, UPPER].mean(axis=1)
		lower_body_errors = per_joint_error_np[:, LOWER].mean(axis=1)

		# Action name mapping (same logic as BaseEval._map_action_name)
		_action_map = mo2cap2_evaluate.config.load_config().actions

		def _map_action(name):
			suffix = re.findall(r'_mixamo_com.*', name)
			if suffix:
				name = name.replace(suffix[0], '')
			return _action_map.get(name, 'All')

		def _build_results_dict(errors_np):
			"""Build per-action results dict matching original format."""
			res = {
				'All': {
					'mpjpe': float(np.mean(errors_np)),
					'std_mpjpe': float(np.std(errors_np)),
					'num_samples': len(errors_np),
				}
			}
			if self.use_action and batch_actions:
				groups = defaultdict(list)
				for i, act in enumerate(batch_actions):
					mapped = _map_action(act)
					groups[mapped].append(errors_np[i])
				for act_name, act_errors in groups.items():
					act_arr = np.array(act_errors)
					res[act_name] = {
						'mpjpe': float(np.mean(act_arr)),
						'std_mpjpe': float(np.std(act_arr)),
						'num_samples': len(act_arr),
					}
			return res

		test_mpjpe = _build_results_dict(full_body_errors)
		test_mpjpe_upper = _build_results_dict(upper_body_errors)
		test_mpjpe_lower = _build_results_dict(lower_body_errors)
		test_mpjpe_per_joint = per_joint_error_np.mean(axis=0)  # (16,)

		mo2cap2_results = {
			"Full Body": test_mpjpe,
			"Upper Body": test_mpjpe_upper,
			"Lower Body": test_mpjpe_lower,
			"Per Joint": test_mpjpe_per_joint
		}

		wandb_results = OrderedDict()
		for k, v in mo2cap2_results.items():
			loss_name = k
			if k == 'Per Joint':
				continue
			for k_, v_ in v.items():
				loss_name += f'_{k_}_mpjpe'
				wandb_results.update({loss_name: v_['mpjpe']})

		eval_results = OrderedDict()
		logger.info(f'Evaluating {self.__class__.__name__}...')
		eval_results.update(wandb_results)

		return eval_results
