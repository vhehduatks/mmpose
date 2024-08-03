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


@DATASETS.register_module(name='Mo2Cap2CocoDataset')
class Mo2Cap2CocoDataset(BaseCocoStyleDataset):
	# METAINFO = None
	# METAPATH = r'C:\Users\user\Documents\GitHub\mmpose\temp_modify\HMD_metainfo\annotation_definitions.json'
	# with open(METAPATH,'r') as _meta:
	# 	METAINFO: dict = json.load(_meta)
	METAINFO: dict = dict(from_file=r'C:\Users\user\Documents\GitHub\mmpose\configs\_base_\datasets\custom_mo2cap2.py')

	MM_TO_M = 1000

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
		# self.index = self._load_index()
		self.test_mode = test_mode
		self.data_root = data_root
		self.chunk_folders = self._get_chunk_folders()

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
		

		# if self.test_mode:
		# 	# save the ann_file into MessageHub for CocoMetric
		# 	message = MessageHub.get_current_instance()
		# 	dataset_name = self.metainfo['dataset_name']
		# 	message.update_info_dict(
		# 		{f'{dataset_name}_ann_file': self.ann_file})

	def _get_chunk_folders(self):
		if self.test_mode:
			res_path = [f for f in os.listdir(self.data_root) if f in ['olek_outdoor','weipeng_studio']]
		else:
			res_path = [f for f in os.listdir(self.data_root) if f.startswith('mo2cap2_chunk_')]
		return res_path

	# def index_db(self):
	# 	return self._index_dir(self.ann_file)
		
	# def _load_index(self):
	# 	idx_path = os.path.join(self.ann_file, 'index.h5')
		
	# 	if os.path.exists(idx_path):
	# 		return self.read_h5(idx_path)

	# 	index = self.index_db()
	# 	self.write_h5(idx_path, index)
	# 	return index

	# def write_h5(self, path, data):
	# 	if '.h5' not in path[-3:]:
	# 		path += '.h5'

	# 	hf = h5py.File(path, 'w')

	# 	if isinstance(data, dict):
	# 		for k, v in data.items():
	# 			if isinstance(v[0], str):
	# 				v = [a.encode('utf8') for a in v]
	# 			hf.create_dataset(k, data=v)
	# 	elif isinstance(data, list):
	# 		hf.create_dataset('val', data=data)
	# 	elif isinstance(data, np.ndarray):
	# 		hf.create_dataset('val', data=data)
	# 	else:
	# 		raise NotImplementedError
	# 	hf.close()

	# def read_h5(self, path):
	# 	if not os.path.isfile(path):
	# 		raise FileNotFoundError()

	# 	data_files = dict()
	# 	h5_data = h5py.File(path)
	# 	tags = list(h5_data.keys())
	# 	for tag in tags:
	# 		tag_data = np.asarray(h5_data[tag]).copy()
	# 		data_files.update({tag: tag_data})
	# 	h5_data.close()

	# 	return data_files

	# def _index_dir(self, path): # png와 json 순서가 다르게 들어감
	# 	indexed_paths = {
	# 		'rgba': [],
	# 		# 'depth': [],
	# 		'frame_data': [],
	# 		# 'segmentation': []
	# 	}

	# 	for root, dirs, files in os.walk(path):
	# 		# if self.test_mode:
	# 		# 	if root.split(os.path.sep)[-1].startswith('olek') or root.split(os.path.sep)[-1].startswith('weipeng'):
	# 		# else:
	# 		# 	if root.split(os.path.sep)[-1].startswith('mo2cap2_'):
	# 		# 		pass

	# 		if root.split(os.path.sep)[-1].startswith('json') or root.split(os.path.sep)[-1].startswith('rgba'):
	# 			for file in files:
	# 				full_path = os.path.join(root, file)
	# 				if file.endswith('.png'):
	# 					indexed_paths['rgba'].append(full_path.encode('utf8'))
	# 				elif file.endswith('.json'):
	# 					indexed_paths['frame_data'].append(full_path.encode('utf8'))


	# 	return indexed_paths


	def parse_data_info(self, _rgba, _frame_data) -> Optional[dict]:
		# JSON 파일 읽기
		# _rgba = _rgba.decode('utf8')
		# _frame_data = _frame_data.decode('utf8')

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

			p2d = np.zeros((15, 2))
			p3d = np.zeros((15, 3))

			joint_names = []
			for key in frame_data.keys():
				if key not in ['action', 'Head']: # keys to skip from json
					joint_names.append(key)


			x_min, y_min = 300,300
			x_max, y_max = 0,0
			# TODO : joint_names 순서랑 metainfo 순서랑 다름. 순서확인할것 joint_names 배열 순서 수정할 것 metainfo 있는걸로
			# 근데 왜 visuall hook에는 예측된 2d 가 정상으로 보이지?
			for jid, joint_name in enumerate(joint_names):
				p2d[jid][0] = frame_data[joint_name]['2d'][0] - 33
				p2d[jid][1] = frame_data[joint_name]['2d'][1]
				# if frame_data[joint_name]['2d'][0] < x_min: x_min = frame_data[joint_name]['2d'][0]
				# if frame_data[joint_name]['2d'][1] < y_min: y_min = frame_data[joint_name]['2d'][1]
				# if frame_data[joint_name]['2d'][0] > x_max: x_max = frame_data[joint_name]['2d'][0]
				# if frame_data[joint_name]['2d'][1] > y_max: y_max = frame_data[joint_name]['2d'][1]
				
				p3d[jid][0] = frame_data[joint_name]['3d'][0]
				p3d[jid][1] = frame_data[joint_name]['3d'][1]
				p3d[jid][2] = frame_data[joint_name]['3d'][2]

	
			
			p3d /= self.MM_TO_M
			
			# bbox = np.array([x_min-10, y_min-10, x_max+10, y_max+10])
			# bbox = np.clip(bbox,0,256)
			# 어차피 mo2cap2 dataset의 testset에는 2차원 keypoint에 대한 정보가 없음. 그냥 이미지 전체가 bbox로 취급
			# TODO : 이거 bbox 가 달라짐(test set 이미지의 크기가1280x1024이고 , trainset의  이미지 크기는 256x256임)
			if self.test_mode:
				#img = img[:, 180:1120, :] # no-crop
				bbox = np.array([180,0,1120,1024]) #xyxy
				# bbox = np.array([0,0,1280,1024]) #xyxy
		
			else:
				bbox = np.array([0,0,256,256])
			bbox = bbox[np.newaxis,:]

		except KeyError as e:
			print(f'Error: key{e}')
			return None

		def extract_and_combine_numbers(file_path):
			# Split the file path by the directory separator
			parts = file_path.split('\\')
			file_part = parts[-1]
			if self.test_mode:
				if 'olek_outdoor' in parts: # fc2_save_2017-10-11-135418-0538
					file_number = file_part.split('-')[-1].split('.')[0].lstrip('0')
					state_number = file_part.split('-')[-2].lstrip('0')
				elif 'weipeng_studio' in parts: # frame_c_0_f_0387
					file_number = file_part.split('_')[-1].split('.')[0].lstrip('0')
					state_number = ''
			else:	
				# Extract the numbers and strip leading zeros
				file_number = file_part.split('_')[-1].split('.')[0].lstrip('0')
				state_number = file_part.split('_')[-2].lstrip('0')

			# Combine the numbers
			try:
				combined_number = int(state_number + file_number)
			except Exception as e:
				print(e)
			return combined_number

		img_id = extract_and_combine_numbers(_frame_data)


		keypoints = p2d
		keypoints_visible = np.ones((1,15),dtype=np.float32)
		keypoint3d = p3d
		keypoints = np.array(keypoints).reshape(1, -1, 2)
		keypoint3d = np.array(keypoint3d).reshape(1, -1, 3)

		area = np.clip((x_max - x_min) * (y_max  - y_min) * 0.53, a_min=1.0, a_max=None)
		area = np.array(area, dtype=np.float32)

		# data_info 딕셔너리 생성
		if self.test_mode:
			data_info = {
				'img_id': img_id,
				'img_path': _rgba,
				'num_keypoints': 15,
				'keypoints': keypoints, # mo2cap test set 에는 2d keypoint가 0 임.
				'keypoints_visible': keypoints_visible,
				'keypoint3d': keypoint3d,
				'action': [frame_data['action']],
				'bbox' : bbox, 
				'bbox_score': np.ones(1, dtype=np.float32),
				'area': area,
				'raw_ann_info':dict(
						id=1,
						image_id=img_id,
						category_id=np.ones(1, dtype=np.float32),
						# bbox=bbox,
						# keypoints=keypoints,
						keypoint3d=keypoint3d,
						iscrowd=0,
						# area=area,
						num_keypoints = 15,
					),
			}
		else:
			data_info = {
				'img_id': img_id,
				'img_path': _rgba,
				'depth_path': None,
				'segmentation_path': None,
				'num_keypoints': 15,
				'keypoints': keypoints,
				'keypoints_visible': keypoints_visible,
				'keypoint3d': keypoint3d,
				'bbox' : bbox, 
				'bbox_score': np.ones(1, dtype=np.float32),
				'area': area,
				'raw_ann_info':dict(
						id=1,
						image_id=img_id,
						category_id=np.ones(1, dtype=np.float32),
						bbox=bbox,
						keypoints=keypoints,
						keypoint3d=keypoint3d,
						iscrowd=0,
						area=area,
						num_keypoints = 15,
					),
			}


		return data_info



	def load_data_list(self) -> List[dict]:
		"""Load data list from COCO annotation file or person detection result
		file."""

				# assert exists(self.ann_file), (
		# 	f'Annotation file `{self.ann_file}`does not exist')
		# temp_path = r'C:\\Users\\user\\Documents\\GitHub\\mmpose\\data\\coco\\annotations\\person_keypoints_test-dev-2017.json'
		# # with get_local_path(self.ann_file) as local_path:
		# self.temp_coco = COCO(temp_path)
		# # set the metainfo about categories, which is a list of dict
		# # and each dict contains the 'id', 'name', etc. about this category
		# if 'categories' in self.temp_coco.dataset:
		# 	self._metainfo['CLASSES'] = self.temp_coco.loadCats(
		# 		self.temp_coco.getCatIds())
			
		person_info = {
			'supercategory': 'person',
			'id': 1,
			'name': 'person',
			'skeleton': [
				[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], 
				[6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], 
				[4, 6], [5, 7]
			],
			'keypoints': [
				'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 
				'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
				'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
			]
		}

		self._metainfo['CLASSES'] = person_info


		file_path_list = []
		data_list = []
		for chunk_folder in self.chunk_folders:
			img_folder = os.path.join(self.data_root, chunk_folder, 'rgba')
			json_folder = os.path.join(self.data_root, chunk_folder, 'json')

			img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])

			for img_file in img_files:
				img_path = os.path.join(img_folder, img_file)
				json_file = img_file.replace('.png', '.json')
				json_path = os.path.join(json_folder, json_file)
				
				if os.path.exists(json_path):
					file_path_list.append({
						'img_path': img_path,
						'ann_path': json_path
					})

		for file_path in file_path_list:
			data_info = self.parse_data_info(file_path['img_path'],file_path['ann_path'])
			if not data_info: continue
			data_list.append(data_info)	 

		return data_list



		# if self.bbox_file:
		# 	data_list = self._load_detection_results()
		# else:
		# 	instance_list, image_list = self._load_annotations()

		# 	if self.data_mode == 'topdown':
		# 		if self.test_mode:
		# 			data_list = instance_list
		# 		else:
		# 			data_list = self._get_topdown_data_infos(instance_list)
		# 	else:
		# 		data_list = self._get_bottomup_data_infos(
		# 			instance_list, image_list)

		