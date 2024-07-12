# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
								 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead

from mmengine.structures import InstanceData

import numpy as np

OptIntSeq = Optional[Sequence[int]]

## A simple yet effective baseline for 3d human pose estimation
'''
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
'''

import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
				linear_size=1024,
				num_stage=2,
				p_dropout=0.5,
				HMD_info = False,
				):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  17 * 2
        # 3d joints
        self.output_size = 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y
##

@MODELS.register_module()
class CustomHeatmapHead(BaseHead):
	"""Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
	(2018). The head is composed of a few deconvolutional layers followed by a
	convolutional layer to generate heatmaps from low-resolution feature maps.

	Args:
		in_channels (int | Sequence[int]): Number of channels in the input
			feature map
		out_channels (int): Number of channels in the output heatmap
		deconv_out_channels (Sequence[int], optional): The output channel
			number of each deconv layer. Defaults to ``(256, 256, 256)``
		deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
			of each deconv layer. Each element should be either an integer for
			both height and width dimensions, or a tuple of two integers for
			the height and the width dimension respectively.Defaults to
			``(4, 4, 4)``
		conv_out_channels (Sequence[int], optional): The output channel number
			of each intermediate conv layer. ``None`` means no intermediate
			conv layer between deconv layers and the final conv layer.
			Defaults to ``None``
		conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
			of each intermediate conv layer. Defaults to ``None``
		final_layer (dict): Arguments of the final Conv2d layer.
			Defaults to ``dict(kernel_size=1)``
		loss (Config): Config of the keypoint loss. Defaults to use
			:class:`KeypointMSELoss`
		decoder (Config, optional): The decoder config that controls decoding
			keypoint coordinates from the network output. Defaults to ``None``
		init_cfg (Config, optional): Config to control the initialization. See
			:attr:`default_init_cfg` for default settings

	.. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
	"""

	_version = 2

	def __init__(self,
				in_channels: Union[int, Sequence[int]],
				out_channels: int,
				deconv_out_channels: OptIntSeq = (256, 256, 256),
				deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
				conv_out_channels: OptIntSeq = None,
				conv_kernel_sizes: OptIntSeq = None,
				final_layer: dict = dict(kernel_size=1),
				loss: ConfigType = dict(
					type='KeypointMSELoss', use_target_weight=True),
				loss_3d: ConfigType = dict(
					type='MSELoss'),
				decoder: OptConfigType = None,
				init_cfg: OptConfigType = None):

		if init_cfg is None:
			init_cfg = self.default_init_cfg

		super().__init__(init_cfg)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.loss_module = MODELS.build(loss)
		self.loss_3d_module = MODELS.build(loss_3d) # 수정

		## 3d baseline
		self.keypoints_3d_module = LinearModel()
		##
		## HMD_info
		self.linear_9_to_34 = nn.Linear(9,34)


		if decoder is not None:
			self.decoder = KEYPOINT_CODECS.build(decoder)
		else:
			self.decoder = None

		if deconv_out_channels:
			if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
					deconv_kernel_sizes):
				raise ValueError(
					'"deconv_out_channels" and "deconv_kernel_sizes" should '
					'be integer sequences with the same length. Got '
					f'mismatched lengths {deconv_out_channels} and '
					f'{deconv_kernel_sizes}')

			self.deconv_layers = self._make_deconv_layers(
				in_channels=in_channels,
				layer_out_channels=deconv_out_channels,
				layer_kernel_sizes=deconv_kernel_sizes,
			)
			in_channels = deconv_out_channels[-1]
		else:
			self.deconv_layers = nn.Identity()

		if conv_out_channels:
			if conv_kernel_sizes is None or len(conv_out_channels) != len(
					conv_kernel_sizes):
				raise ValueError(
					'"conv_out_channels" and "conv_kernel_sizes" should '
					'be integer sequences with the same length. Got '
					f'mismatched lengths {conv_out_channels} and '
					f'{conv_kernel_sizes}')

			self.conv_layers = self._make_conv_layers(
				in_channels=in_channels,
				layer_out_channels=conv_out_channels,
				layer_kernel_sizes=conv_kernel_sizes)
			in_channels = conv_out_channels[-1]
		else:
			self.conv_layers = nn.Identity()

		if final_layer is not None:
			cfg = dict(
				type='Conv2d',
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1)
			cfg.update(final_layer)
			self.final_layer = build_conv_layer(cfg)
		else:
			self.final_layer = nn.Identity()

		# Register the hook to automatically convert old version state dicts
		self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

	def _make_conv_layers(self, in_channels: int,
						  layer_out_channels: Sequence[int],
						  layer_kernel_sizes: Sequence[int]) -> nn.Module:
		"""Create convolutional layers by given parameters."""

		layers = []
		for out_channels, kernel_size in zip(layer_out_channels,
											 layer_kernel_sizes):
			padding = (kernel_size - 1) // 2
			cfg = dict(
				type='Conv2d',
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				stride=1,
				padding=padding)
			layers.append(build_conv_layer(cfg))
			layers.append(nn.BatchNorm2d(num_features=out_channels))
			layers.append(nn.ReLU(inplace=True))
			in_channels = out_channels

		return nn.Sequential(*layers)

	def _make_deconv_layers(self, in_channels: int,
							layer_out_channels: Sequence[int],
							layer_kernel_sizes: Sequence[int]) -> nn.Module:
		"""Create deconvolutional layers by given parameters."""

		layers = []
		for out_channels, kernel_size in zip(layer_out_channels,
											 layer_kernel_sizes):
			if kernel_size == 4:
				padding = 1
				output_padding = 0
			elif kernel_size == 3:
				padding = 1
				output_padding = 1
			elif kernel_size == 2:
				padding = 0
				output_padding = 0
			else:
				raise ValueError(f'Unsupported kernel size {kernel_size} for'
								 'deconvlutional layers in '
								 f'{self.__class__.__name__}')
			cfg = dict(
				type='deconv',
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				stride=2,
				padding=padding,
				output_padding=output_padding,
				bias=False)
			layers.append(build_upsample_layer(cfg))
			layers.append(nn.BatchNorm2d(num_features=out_channels))
			layers.append(nn.ReLU(inplace=True))
			in_channels = out_channels

		return nn.Sequential(*layers)

	@property
	def default_init_cfg(self):
		init_cfg = [
			dict(
				type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
			dict(type='Constant', layer='BatchNorm2d', val=1)
		]
		return init_cfg

	def forward(self, feats: Tuple[Tensor]) -> Tensor:
		"""Forward the network. The input is multi scale feature maps and the
		output is the heatmap.

		Args:
			feats (Tuple[Tensor]): Multi scale feature maps.

		Returns:
			Tensor: output heatmap.
		"""
		x = feats[-1]

		x = self.deconv_layers(x)
		x = self.conv_layers(x)
		x = self.final_layer(x)

		return x

	def decode(self, batch_outputs: Union[Tensor,Tuple[Tensor]], batch_data_samples: OptSampleList) -> InstanceList:
		"""Decode keypoints from outputs.

		Args:
			batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
				a data batch

		Returns:
			List[InstanceData]: A list of InstanceData, each contains the
			decoded pose information of the instances of one data sample.
		"""

		def _pack_and_call(args, func):
			if not isinstance(args, tuple):
				args = (args, )
			return func(*args)

		if self.decoder is None:
			raise RuntimeError(
				f'The decoder has not been set in {self.__class__.__name__}. '
				'Please set the decoder configs in the init parameters to '
				'enable head methods `head.predict()` and `head.decode()`')

		if self.decoder.support_batch_decoding:
			batch_keypoints, batch_scores = _pack_and_call(
				batch_outputs, self.decoder.batch_decode)
			if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
				batch_scores, batch_visibility = batch_scores
			else:
				batch_visibility = [None] * len(batch_keypoints)

		else:
			batch_output_np = to_numpy(batch_outputs, unzip=True)
			batch_keypoints = []
			batch_scores = []
			batch_visibility = []
			for outputs in batch_output_np:
				keypoints, scores = _pack_and_call(outputs,
												   self.decoder.decode)
				batch_keypoints.append(keypoints)
				if isinstance(scores, tuple) and len(scores) == 2:
					batch_scores.append(scores[0])
					batch_visibility.append(scores[1])
				else:
					batch_scores.append(scores)
					batch_visibility.append(None)

		preds = []



		## HMD_info
		
		HMD_info = torch.cat([
			d.gt_instance_labels.hmd_info for d in batch_data_samples
		])
		HMD_info = HMD_info.flatten(start_dim=1)
		HMD_info = self.linear_9_to_34(HMD_info.to(torch.float32))
		

		## 3d baseline		
		batch_3d_keypoints = torch.tensor(batch_keypoints,device=batch_outputs.device).squeeze().view(-1,34)
		
		batch_3d_keypoints = self.keypoints_3d_module(batch_3d_keypoints+HMD_info)
		# batch_3d_keypoints = self.keypoints_3d_module(batch_3d_keypoints)
		##

		for keypoints, keypoint_3d, scores, visibility in zip(batch_keypoints, batch_3d_keypoints, batch_scores,
												 batch_visibility):
			keypoint_3d = keypoint_3d.unsqueeze(dim=0)
			pred = InstanceData(keypoints=keypoints, keypoint_scores=scores, keypoint_3d=keypoint_3d)
			if visibility is not None:
				pred.keypoints_visible = visibility
			preds.append(pred)

		return preds,batch_3d_keypoints

	def predict(self,
				feats: Features,
				batch_data_samples: OptSampleList,
				test_cfg: ConfigType = {}) -> Predictions:
		"""Predict results from features.

		Args:
			feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
				features (or multiple multi-stage features in TTA)
			batch_data_samples (List[:obj:`PoseDataSample`]): The batch
				data samples
			test_cfg (dict): The runtime config for testing process. Defaults
				to {}

		Returns:
			Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
			``test_cfg['output_heatmap']==True``, return both pose and heatmap
			prediction; otherwise only return the pose prediction.

			The pose prediction is a list of ``InstanceData``, each contains
			the following fields:

				- keypoints (np.ndarray): predicted keypoint coordinates in
					shape (num_instances, K, D) where K is the keypoint number
					and D is the keypoint dimension
				- keypoint_scores (np.ndarray): predicted keypoint scores in
					shape (num_instances, K)

			The heatmap prediction is a list of ``PixelData``, each contains
			the following fields:

				- heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
		"""

		if test_cfg.get('flip_test', False):
			# TTA: flip test -> feats = [orig, flipped]
			assert isinstance(feats, list) and len(feats) == 2
			flip_indices = batch_data_samples[0].metainfo['flip_indices']
			_feats, _feats_flip = feats
			_batch_heatmaps = self.forward(_feats)
			_batch_heatmaps_flip = flip_heatmaps(
				self.forward(_feats_flip),
				flip_mode=test_cfg.get('flip_mode', 'heatmap'),
				flip_indices=flip_indices,
				shift_heatmap=test_cfg.get('shift_heatmap', False))
			batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
		else:
			batch_heatmaps = self.forward(feats)


		preds,_ = self.decode(batch_heatmaps, batch_data_samples)

		if test_cfg.get('output_heatmaps', False):
			pred_fields = [
				PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
			]
			return preds, pred_fields
		else:
			return preds

	def loss(self,
			 feats: Tuple[Tensor],
			 batch_data_samples: OptSampleList,
			 train_cfg: ConfigType = {}) -> dict:
		"""Calculate losses from a batch of inputs and data samples.

		Args:
			feats (Tuple[Tensor]): The multi-stage features
			batch_data_samples (List[:obj:`PoseDataSample`]): The batch
				data samples
			train_cfg (dict): The runtime config for training process.
				Defaults to {}

		Returns:
			dict: A dictionary of losses.
		"""
		pred_fields = self.forward(feats)
		gt_heatmaps = torch.stack(
			[d.gt_fields.heatmaps for d in batch_data_samples])
		keypoint_weights = torch.cat([
			d.gt_instance_labels.keypoint_weights for d in batch_data_samples
		])

		## 3d baseline
		gt_keypoint_3d = torch.cat([
			d.gt_instance_labels.keypoint3d for d in batch_data_samples
		])



		gt_keypoint_3d = gt_keypoint_3d.view(-1,51)

		
		_,pred_batch_3d_keypoints = self.decode(pred_fields,batch_data_samples)
		loss_3d = self.loss_3d_module(pred_batch_3d_keypoints.to(torch.double),gt_keypoint_3d.to(torch.double))
		##

		# calculate losses
		losses = dict()
		loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
		
		## 3d baseline
		losses.update(loss_kpt3d = loss_3d)
		##
		
		losses.update(loss_kpt=loss)

		# calculate accuracy
		if train_cfg.get('compute_acc', True):
			_, avg_acc, _ = pose_pck_accuracy(
				output=to_numpy(pred_fields),
				target=to_numpy(gt_heatmaps),
				mask=to_numpy(keypoint_weights) > 0)

			acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
			losses.update(acc_pose=acc_pose)

		return losses

	def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
								  **kwargs):
		"""A hook function to convert old-version state dict of
		:class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
		compatible format of :class:`HeatmapHead`.

		The hook will be automatically registered during initialization.
		"""
		version = local_meta.get('version', None)
		if version and version >= self._version:
			return

		# convert old-version state dict
		keys = list(state_dict.keys())
		for _k in keys:
			if not _k.startswith(prefix):
				continue
			v = state_dict.pop(_k)
			k = _k[len(prefix):]
			# In old version, "final_layer" includes both intermediate
			# conv layers (new "conv_layers") and final conv layers (new
			# "final_layer").
			#
			# If there is no intermediate conv layer, old "final_layer" will
			# have keys like "final_layer.xxx", which should be still
			# named "final_layer.xxx";
			#
			# If there are intermediate conv layers, old "final_layer"  will
			# have keys like "final_layer.n.xxx", where the weights of the last
			# one should be renamed "final_layer.xxx", and others should be
			# renamed "conv_layers.n.xxx"
			k_parts = k.split('.')
			if k_parts[0] == 'final_layer':
				if len(k_parts) == 3:
					assert isinstance(self.conv_layers, nn.Sequential)
					idx = int(k_parts[1])
					if idx < len(self.conv_layers):
						# final_layer.n.xxx -> conv_layers.n.xxx
						k_new = 'conv_layers.' + '.'.join(k_parts[1:])
					else:
						# final_layer.n.xxx -> final_layer.xxx
						k_new = 'final_layer.' + k_parts[2]
				else:
					# final_layer.xxx remains final_layer.xxx
					k_new = k
			else:
				k_new = k

			state_dict[prefix + k_new] = v
