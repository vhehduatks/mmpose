# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
# from ..base import BaseCocoStyleDataset
from mmengine.dataset import BaseDataset

import copy
import os.path as osp
from copy import deepcopy
from itertools import chain, filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


import functools
import gc
import logging
import pickle
from collections.abc import Mapping


from torch.utils.data import Dataset

from mmengine.config import Config
from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS
from mmengine.utils import is_abs

##
import os
import re
import h5py
import json
from .config import config

@DATASETS.register_module(name='CustomEgoposeDataset')
class CustomEgoposeDataset(BaseDataset):

	ROOT_DIRS = ['rgba','depth','json','objectId']
	CM_TO_M = 100

	def __init__(self, 
			  transform=None,
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
		"""Init class

		Arguments:
			db_path {str} -- path to set
			set_type {SetType} -- set

		Keyword Arguments:
			transform {BaseTransform} -- transformation to apply to data (default: {None})
		"""
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
		
		
		
	def index_db(self):
		return self._index_dir(self.ann_file)
	
	def _load_index(self):
		"""Get indexed set. If the set has already been
		indexed, load the file, otherwise index it and save cache.

		Returns:
			dict -- index set
		"""

		idx_path = os.path.join(self.ann_file, 'index.h5')
		
		if os.path.exists(idx_path):
			return self.read_h5(idx_path)

		index = self.index_db()
		self.write_h5(idx_path, index)
		return index

	def get_files(self, path, formats=None):
		"""Get files contained in path

		Arguments:
			path {str} -- path

		Keyword Arguments:
			formats {str/list} -- file formats; if None take all (default: {None})

		Returns:
			list -- file names
			list -- file paths
		"""

		if formats:

			if isinstance(formats, str):
				formats = [formats]
			else:
				assert isinstance(formats, list)

			names = []
			paths = []

			for f_format in formats:
				files = [f for f in os.listdir(path)
						if re.match(r'.*\.{}'.format(f_format), f)]

				files.sort()
				names.extend(files)
				paths.extend([os.path.join(path, f) for f in files])

			return names, paths

		names = os.listdir(path)
		names.sort()
		paths = [os.path.join(path, f) for f in names]

		return names, paths	

	def write_h5(self, path, data):
		"""Write h5 file

		Arguments:
			path {str} -- file path where to save the data
			data {seriaizable} -- data to be saved

		Raises:
			NotImplementedError -- non serializable data to save
		"""

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
		"""Load data from h5 file

		Arguments:
			path {str} -- file path

		Raises:
			FileNotFoundError -- Path not pointing to a file

		Returns:
			dict -- dictionary containing the data
		"""

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


	def get_subdirs(self, path):
		"""Get directories contained in path

		Arguments:
			path {str} -- path

		Returns:
			list -- directory names
			list -- directory paths
		"""

		try:
			names = os.walk(path).next()[1]
		except AttributeError:
			names = next(os.walk(path))[1]

		names.sort()
		dir_paths = [os.path.join(path, n) for n in names]

		return names, dir_paths

	def _index_dir(self, path):
		"""Recursively add paths to the set of
		indexed files

		Arguments:
			path {str} -- folder path

		Returns:
			dict -- indexed files per root dir
		"""

		indexed_paths = dict()
		sub_dirs, _ = self.get_subdirs(path)
		if set(self.ROOT_DIRS) <= set(sub_dirs):

			# get files from subdirs
			n_frames = -1

			# let's extract the rgba and json data per frame
			for sub_dir in self.ROOT_DIRS:
				d_path = os.path.join(path, sub_dir)
				_, paths = self.get_files(d_path)

				if n_frames < 0:
					n_frames = len(paths)
				else:
					if len(paths) != n_frames:
						raise('Frames info in {} not matching other passes'.format(d_path))

				encoded = [p.encode('utf8') for p in paths]
				indexed_paths.update({sub_dir: encoded})

			return indexed_paths

		# initialize indexed_paths
		for sub_dir in self.ROOT_DIRS:
			indexed_paths.update({sub_dir: []})

		# check subdirs of path and merge info
		for sub_dir in sub_dirs:
			indexed = self._index_dir(os.path.join(path, sub_dir))

			for r_dir in self.ROOT_DIRS:
				indexed_paths[r_dir].extend(indexed[r_dir])

		return indexed_paths



	def load_data_list(self) -> List[dict]:
		"""Load annotations from an annotation file named as ``self.ann_file``

		If the annotation file does not follow `OpenMMLab 2.0 format dataset
		<https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
		The subclass must override this method for load annotations. The meta
		information of annotation file will be overwritten :attr:`METAINFO`
		and ``metainfo`` argument of constructor.

		Returns:
			list[dict]: A list of annotation.
		"""  # noqa: E501
		# `self.ann_file` denotes the absolute annotation file path if
		# `self.root=None` or relative path if `self.root=/path/to/data/`.
		# annotations = load(self.ann_file)
		# if not isinstance(annotations, dict):
		# 	raise TypeError(f'The annotations loaded from annotation file '
		# 					f'should be a dict, but got {type(annotations)}!')
		# if 'data_list' not in annotations or 'metainfo' not in annotations:
		# 	raise ValueError('Annotation must have data_list and metainfo '
		# 					'keys')
		# metainfo = annotations['metainfo']
		# raw_data_list = annotations['data_list']

		# Meta information load from annotation file will not influence the
		# existed meta information load from `BaseDataset.METAINFO` and
		# `metainfo` arguments defined in constructor.
		# for k, v in metainfo.items():
		# 	self._metainfo.setdefault(k, v)

		# load and parse data_infos.
		data_list = []
		for img_path, depth_path, seg_path, json_path in zip(self.index['rgba'], 
													   self.index['depth'], 
													   self.index['objectId'], 
													   self.index['json']):
			
			# parse raw data information to target format
			data_info = self.parse_data_info(
				img_path.decode('utf8'), 
				depth_path.decode('utf8'), 
				seg_path.decode('utf8'), 
				json_path.decode('utf8')
				)
			
			if isinstance(data_info, dict):
				# For image tasks, `data_info` should information if single
				# image, such as dict(img_path='xxx', width=360, ...)
				data_list.append(data_info)
			elif isinstance(data_info, list):
				# For video tasks, `data_info` could contain image
				# information of multiple frames, such as
				# [dict(video_path='xxx', timestamps=...),
				#  dict(video_path='xxx', timestamps=...)]
				for item in data_info:
					if not isinstance(item, dict):
						raise TypeError('data_info must be list of dict, but '
										f'got {type(item)}')
				data_list.extend(data_info)
			else:
				raise TypeError('data_info should be a dict or list of dict, '
								f'but got {type(data_info)}')

		return data_list
	
	def parse_data_info(self, img_path, depth_path, seg_path, json_path) -> Union[dict, List[dict]]:
		"""Parse raw annotation to target format.

		This method should return dict or list of dict. Each dict or list
		contains the data information of a training sample. If the protocol of
		the sample annotations is changed, this function can be overridden to
		update the parsing logic while keeping compatibility.

		Args:
			raw_data_info (dict): Raw data information load from ``ann_file``

		Returns:
			list or list[dict]: Parsed annotation.
		"""
		with open(json_path, 'r') as in_file:
			data = json.load(in_file)
		joint_names = {j['name'].replace('mixamorig:', ''): jid for jid, j in enumerate(data['joints'])}
		p2d_orig = np.array(data['pts2d_fisheye']).T
		p3d_orig = np.array(data['pts3d_fisheye']).T
		kpt3d = np.empty([len(config.skel_16), 3], dtype=p3d_orig.dtype)
		kpt = np.empty([len(config.skel_15), 2], dtype=p2d_orig.dtype)



		for jid, j in enumerate(config.skel_15.keys()):
			kpt[jid] = p2d_orig[joint_names[j]]
		for jid, j in enumerate(config.skel_16.keys()):	
			kpt3d[jid] = p3d_orig[joint_names[j]]

		kpt3d /= self.CM_TO_M
		kpt3d = np.expand_dims(kpt3d, axis=0)
		kpt = np.expand_dims(kpt,axis=0)
		raw_data_info = {
			'img_path' : img_path,
			'depth_path' : depth_path,
			'seg_path' : seg_path,
			'keypoints' : kpt,
			'keypoints_3d' : kpt3d,
			'keypoints_visible' : np.ones((1,16),dtype=np.float32),
		}

		# data_info = {
		# 	'img_id': ann['image_id'],
		# 	'img_path': img_path,
		# 	'bbox': bbox,
		# 	'bbox_score': np.ones(1, dtype=np.float32),
		# 	'num_keypoints': num_keypoints,
		# 	'keypoints': keypoints,
		# 	'keypoints_visible': keypoints_visible,
		# 	'iscrowd': ann.get('iscrowd', 0),
		# 	'segmentation': ann.get('segmentation', None),
		# 	'id': ann['id'],
		# 	'category_id': ann['category_id'],
		# 	# store the raw annotation of the instance'
		# 	# it is useful for evaluation without providing ann_file
		# 	'raw_ann_info': copy.deepcopy(ann),
		# }
		return raw_data_info
	
	def filter_data(self) -> List[dict]:
		"""Filter annotations according to filter_cfg. Defaults return all
		``data_list``.

		If some ``data_list`` could be filtered according to specific logic,
		the subclass should override this method.

		Returns:
			list[int]: Filtered results.
		"""
		return self.data_list