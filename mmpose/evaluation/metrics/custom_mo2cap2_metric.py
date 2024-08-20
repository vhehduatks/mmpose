# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

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
class CustomMo2Cap2Metric(BaseMetric):
	default_prefix: Optional[str] = 'mo2cap2'

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
		self.eval_body = mo2cap2_evaluate.EvalBody(mode='mo2cap2')
		self.eval_upper = mo2cap2_evaluate.EvalUpperBody(mode='mo2cap2')
		self.eval_lower = mo2cap2_evaluate.EvalLowerBody(mode='mo2cap2')
		self.eval_per_joint = mo2cap2_evaluate.EvalPerJoint(mode='mo2cap2')
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
			pred['id'] = data_sample['img_id']
			pred['img_id'] = data_sample['img_id']

			pred['keypoints'] = keypoints
			pred['keypoint_scores'] = keypoint_scores
			pred['category_id'] = data_sample.get('category_id', 1)

			## 3d baseline
			pred['keypoint3d'] = data_sample['pred_instances']['keypoint_3d']
			##


			# parse gt
			gt = dict()
			if self.coco is None:
				gt['width'] = data_sample['ori_shape'][1]
				gt['height'] = data_sample['ori_shape'][0]
				gt['img_id'] = data_sample['img_id']
				if self.iou_type == 'keypoints_crowd':
					assert 'crowd_index' in data_sample, \
						'`crowd_index` is required when `self.iou_type` is ' \
						'`keypoints_crowd`'
					gt['crowd_index'] = data_sample['crowd_index']
				assert 'raw_ann_info' in data_sample, \
					'The row ground truth annotations are required for ' \
					'evaluation when `ann_file` is not provided'
				anns = data_sample['raw_ann_info']
				gt['raw_ann_info'] = anns if isinstance(anns, list) else [anns]
			## 3d baseline
			gt['keypoint3d'] = data_sample['gt_instance_labels']['keypoint3d']
			##

			## mo2cap2
			if self.use_action:
				gt['action'] = data_sample['gt_instances']['action'][0]
			##

## TODO metric 수정할 것 gt , pred 둘다 data_sample에 있ㅇ므
			# add converted result to the results list
			self.results.append((pred, gt))



#TODO : 이거 egostan evaluate compute_metrics 사용해서 배치당 에러 dict에 저장 (process에서 pred,gt 정리해서 result에 저장후 이걸 compute_metrics에서 계산함)
	def compute_metrics(self, results: list) -> Dict[str, float]:

		logger: MMLogger = MMLogger.get_current_instance()

		# split prediction and gt list
		preds, gts = zip(*results)
		
		## 3d baseline
		pred_batch_3d_keypoints = []
		gt_batch_keypoint_3d = []
		# if self.use_action:
		batch_actions = []

		for pred_, gt_ in zip(preds,gts):
			pred_batch_3d_keypoints.append(pred_['keypoint3d'])
			gt_batch_keypoint_3d.append(gt_['keypoint3d'])
			if self.use_action:
				batch_actions.append(gt_['action'])
		
		pred_batch_3d_keypoints = torch.stack(pred_batch_3d_keypoints).squeeze()
		gt_batch_keypoint_3d = torch.stack(gt_batch_keypoint_3d).squeeze()

		##

		##mo2cap2 baseline
		self.eval_body.eval(pred_batch_3d_keypoints, gt_batch_keypoint_3d, batch_actions, use_action_ = self.use_action)
		self.eval_upper.eval(pred_batch_3d_keypoints, gt_batch_keypoint_3d, batch_actions, use_action_ = self.use_action)
		self.eval_lower.eval(pred_batch_3d_keypoints, gt_batch_keypoint_3d, batch_actions, use_action_ = self.use_action)
		self.eval_per_joint.eval(pred_batch_3d_keypoints, gt_batch_keypoint_3d)

		test_mpjpe = self.eval_body.get_results()
		test_mpjpe_upper = self.eval_upper.get_results()
		test_mpjpe_lower = self.eval_lower.get_results()
		test_mpjpe_per_joint = self.eval_per_joint.get_results()


		'''
		coco/Full Body: {
		'All': {'mpjpe': 206.74012547793745, 'std_mpjpe': 2.3492799070270713, 'num_samples': 34},
		'walking': {'mpjpe': 206.74012547793745, 'std_mpjpe': 2.3492799070270713, 'num_samples': 34}
		}
		coco/Upper Body: {'All': {'mpjpe': 98.853390965369, 'std_mpjpe': 1.8284284966864535, 'num_samples': 34}, 'walking': {'mpjpe': 98.853390965369, 'std_mpjpe': 1.8284284966864535, 'num_samples': 34}}  coco/Lower Body: {'All': {'mpjpe': 301.1410181764348, 'std_mpjpe': 4.4306975170390555, 'num_samples': 34}, 'walking': {'mpjpe': 301.1410181764348, 'std_mpjpe': 4.4306975170390555, 'num_samples': 34}} 
		coco/Per Joint: 
		[ 89.73435511  55.46927861  56.26292247  93.79446133 121.92435555
		146.15558961 128.6327741  145.93864229 268.94727702 381.7553911
		398.04040747 153.16333653 276.8626541  384.93890462 399.48153228]
		'''

		mo2cap2_results = {
			"Full Body": test_mpjpe,
			"Upper Body": test_mpjpe_upper,
			"Lower Body": test_mpjpe_lower,
			"Per Joint": test_mpjpe_per_joint
		}
		# TODO : 전체결과는 저장하고 mpjpe 는 wandb로
		# mo2cap2_evaluate.create_results_csv(mo2cap2_results)
		##
		wandb_results = OrderedDict()
		for k,v in mo2cap2_results.items():
			loss_name = k
			if k == 'Per Joint': continue
			for k_,v_ in v.items():
				loss_name += f'_{k_}_mpjpe'
				wandb_results.update({loss_name:v_['mpjpe']})

		# evaluation results
		eval_results = OrderedDict()
		logger.info(f'Evaluating {self.__class__.__name__}...')
		# info_str = self._do_python_keypoint_eval(outfile_prefix)
		# name_value = OrderedDict(info_str)
		# eval_results.update(name_value)
		eval_results.update(wandb_results)

		# if tmp_dir is not None:
		# 	tmp_dir.cleanup()
		return eval_results
