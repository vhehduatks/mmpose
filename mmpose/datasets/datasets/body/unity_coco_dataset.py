# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from copy import deepcopy
from itertools import chain, filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.logging import MessageHub
from mmengine.utils import is_list_of
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..utils import parse_pose_metainfo
import os
import re
import h5py
import json


@DATASETS.register_module(name='UnityCocoDataset')
class UnityCocoDataset(BaseCocoStyleDataset):
	# METAINFO = None
	# METAPATH = r'C:\Users\user\Documents\GitHub\mmpose\temp_modify\HMD_metainfo\annotation_definitions.json'
	# with open(METAPATH,'r') as _meta:
	# 	METAINFO: dict = json.load(_meta)
	METAINFO: dict = dict(from_file='configs/_base_/datasets/coco.py')

	def __init__(self,
				ann_file: str = '',
				bbox_file: Optional[str] = None,
				data_mode: str = 'topdown',
				metainfo: Optional[dict] = None,
				data_root: Optional[str] = None,
				data_prefix: dict = dict(img=''),
				filter_cfg: Optional[dict] = None,
				indices: Optional[Union[int, Sequence[int]]] = None,
				serialize_data: bool = True,
				pipeline: List[Union[dict, Callable]] = [],
				test_mode: bool = False,
				lazy_init: bool = False,
				max_refetch: int = 1000,
				sample_interval: int = 1):

		if data_mode not in {'topdown', 'bottomup'}:
			raise ValueError(
				f'{self.__class__.__name__} got invalid data_mode: '
				f'{data_mode}. Should be "topdown" or "bottomup".')
		self.data_mode = data_mode

		if bbox_file:
			if self.data_mode != 'topdown':
				raise ValueError(
					f'{self.__class__.__name__} is set to {self.data_mode}: '
					'mode, while "bbox_file" is only '
					'supported in topdown mode.')

			if not test_mode:
				raise ValueError(
					f'{self.__class__.__name__} has `test_mode==False` '
					'while "bbox_file" is only '
					'supported when `test_mode==True`.')
		self.bbox_file = bbox_file
		self.sample_interval = sample_interval

		self.ann_file = ann_file
		self.index = self._load_index()

		super().__init__(
			ann_file=ann_file,
			metainfo=metainfo,
			data_root=data_root,
			data_prefix=data_prefix,
			filter_cfg=filter_cfg,
			indices=indices,
			serialize_data=serialize_data,
			pipeline=pipeline,
			test_mode=test_mode,
			lazy_init=lazy_init,
			max_refetch=max_refetch)

		

		if self.test_mode:
			# save the ann_file into MessageHub for CocoMetric
			message = MessageHub.get_current_instance()
			dataset_name = self.metainfo['dataset_name']
			message.update_info_dict(
				{f'{dataset_name}_ann_file': self.ann_file})


	def index_db(self):
		return self._index_dir(self.ann_file)
		
	def _load_index(self):
		idx_path = os.path.join(self.ann_file, 'index.h5')
		
		if os.path.exists(idx_path):
			return self.read_h5(idx_path)

		index = self.index_db()
		self.write_h5(idx_path, index)
		return index

	def write_h5(self, path, data):
		if '.h5' not in path[-3:]:
			path += '.h5'

		hf = h5py.File(path, 'w')

		if isinstance(data, dict):
			for k, v in data.items():
				if isinstance(v[0], str):
					v = [a.encode('utf8') for a in v]
				hf.create_dataset(k, data=v)
		elif isinstance(data, list):
			hf.create_dataset('val', data=data)
		elif isinstance(data, np.ndarray):
			hf.create_dataset('val', data=data)
		else:
			raise NotImplementedError
		hf.close()

	def read_h5(self, path):
		if not os.path.isfile(path):
			raise FileNotFoundError()

		data_files = dict()
		h5_data = h5py.File(path)
		tags = list(h5_data.keys())
		for tag in tags:
			tag_data = np.asarray(h5_data[tag]).copy()
			data_files.update({tag: tag_data})
		h5_data.close()

		return data_files

	def _index_dir(self, path):
		indexed_paths = {
			'rgba': [],
			'depth': [],
			'frame_data': [],
			'segmentation': []
		}

		for root, dirs, files in os.walk(path):
			if root.split(os.path.sep)[-1].startswith('sequence.'):
				for file in files:
					full_path = os.path.join(root, file)
					if file.endswith('camera.png'):
						indexed_paths['rgba'].append(full_path.encode('utf8'))
					elif file.endswith('Depth.exr'):
						indexed_paths['depth'].append(full_path.encode('utf8'))
					elif file.endswith('frame_data.json'):
						indexed_paths['frame_data'].append(full_path.encode('utf8'))
					elif file.endswith('instance segmentation.png'):
						indexed_paths['segmentation'].append(full_path.encode('utf8'))

		return indexed_paths


	def parse_data_info(self, _rgba, _depth, _segmentation, _frame_data) -> Optional[dict]:
		# JSON 파일 읽기
		_rgba = _rgba.decode('utf8')
		_depth = _depth.decode('utf8')
		_segmentation = _segmentation.decode('utf8')
		_frame_data = _frame_data.decode('utf8')
		try:
			with open(_frame_data, 'r') as f:
				frame_data = json.load(f)
		except FileNotFoundError:
			print(f"Error: File not found - {_frame_data}")
			return None
		except json.JSONDecodeError:
			print(f"Error: Invalid JSON format in file - {_frame_data}")
			return None
		try:
			for _cap in frame_data['captures']:
				if isinstance(_cap, dict):
					if isinstance(_cap['annotations'], list):
						keypoints_info = None
						keypoint3d_info = None
						bbox_info = None

						for _ann in _cap['annotations']:
							if isinstance(_ann, dict):
								if _ann.get('@type', '').endswith('KeypointAnnotation'):
									keypoints_info = _ann.get('values', [{}])[0].get('keypoints')
								elif _ann.get('@type', '').endswith('Keypoint3dAnnotation'):
									keypoint3d_info = _ann.get('keypoints', [{}])[0].get('keypoints')
								elif _ann.get('@type', '').endswith('BoundingBox2DAnnotation'):
									bbox_values = _ann.get('values', [{}])[0]
									origin = bbox_values.get('origin', [])
									dimension = bbox_values.get('dimension', [])
									if origin and dimension:
										bbox_info = origin + dimension  # This will give [x, y, w, h]

						# Check if we found all required annotations
						assert len(keypoints_info) and len(keypoint3d_info) and len(bbox_info)

						keypoints_info=keypoints_info
						keypoint3d_info=keypoint3d_info
						bbox_info=bbox_info



			x_min, y_min = bbox_info[:2]
			w, h = bbox_info[2:]
			x_max, y_max = x_min + w, y_min + h
			bbox = np.array([x_min, y_min, x_max, y_max])
			bbox = bbox[np.newaxis,:]

		except KeyError as e:
			print(f'Error: key{e}')
			return None

		# 사용할 joint_name 목록 생성
		valid_joint_names = []
		for k,v in self.metainfo['keypoint_id2name'].items():
			if v=='nose' : valid_joint_names.append(v)
			elif '_' in v :
				temp_joint_ = v.split('_')
				valid_joint_names.append(temp_joint_[-1]+'_'+temp_joint_[0])
		assert valid_joint_names 
		# 키포인트 및 가시성 정보 생성
		keypoints = []
		keypoints_visible = []
		keypoint3d = []
		num_keypoints = 0

		for kp in keypoints_info:
			if kp['state'] != 0:
				keypoints.extend(kp['location'])
				keypoints_visible.append(1)
				num_keypoints += 1
			else:
				keypoints.extend([0, 0])
				keypoints_visible.append(0)

		# 3D 키포인트 정보 생성
		for kp in keypoint3d_info:
			if kp['label'] in valid_joint_names:
				keypoint3d.extend(kp['location'])

		keypoints = np.array(keypoints).reshape(1, -1, 2)
		keypoints_visible = np.array(keypoints_visible).reshape(1, -1)
		keypoint3d = np.array(keypoint3d).reshape(1, -1, 3)

		area = np.clip((x_max - x_min) * (y_max  - y_min) * 0.53, a_min=1.0, a_max=None)
		area = np.array(area, dtype=np.float32)

		# data_info 딕셔너리 생성
		data_info = {
			'img_id': frame_data['frame'],
			'img_path': _rgba,
			'depth_path': _depth,
			'segmentation_path': _segmentation,
			'num_keypoints': num_keypoints,
			'keypoints': keypoints,
			'keypoints_visible': keypoints_visible,
			'keypoint3d': keypoint3d,
			'bbox' : bbox, 
			'bbox_score': np.ones(1, dtype=np.float32),
			'area': area,
			'raw_ann_info':dict(
					id=1,
					image_id=frame_data['frame'],
					category_id=np.ones(1, dtype=np.float32),
					bbox=bbox,
					keypoints=keypoints,
					keypoint3d=keypoint3d,
					iscrowd=0,
					area=area,
					num_keypoints = num_keypoints,
				),
		}

		return data_info

	def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
		"""Load data from annotations in COCO format."""

		assert exists(self.ann_file), (
			f'Annotation file `{self.ann_file}`does not exist')
		temp_path = r'C:\\Users\\user\\Documents\\GitHub\\mmpose\\data\\coco\\annotations\\person_keypoints_test-dev-2017.json'
		# with get_local_path(self.ann_file) as local_path:
		self.temp_coco = COCO(temp_path)
		# set the metainfo about categories, which is a list of dict
		# and each dict contains the 'id', 'name', etc. about this category
		if 'categories' in self.temp_coco.dataset:
			self._metainfo['CLASSES'] = self.temp_coco.loadCats(
				self.temp_coco.getCatIds())


		instance_list = []
		image_list = []

		for _rgba, _depth, _segmentation, _frame_data in zip(self.index['rgba'],
													   self.index['depth'],
													   self.index['segmentation'],
													   self.index['frame_data']):
			instance_info = self.parse_data_info(_rgba, _depth, _segmentation, _frame_data)

				# skip invalid instance annotation.
			if not instance_info:
				continue

			instance_list.append(instance_info)			 

		return instance_list, image_list

	def load_data_list(self) -> List[dict]:
		"""Load data list from COCO annotation file or person detection result
		file."""

		if self.bbox_file:
			data_list = self._load_detection_results()
		else:
			instance_list, image_list = self._load_annotations()

			if self.data_mode == 'topdown':
				data_list = self._get_topdown_data_infos(instance_list)
			else:
				data_list = self._get_bottomup_data_infos(
					instance_list, image_list)

		return data_list