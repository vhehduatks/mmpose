# Copyright (c) OpenMMLab. All rights reserved.
"""Baseline head with configurable HMD info size.

This head extends CustomxRegoposeBaselinel1 with a configurable hmd_info_size
parameter to support enhanced HMD info (10, 11, or 12 dimensions instead of 9).
"""
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
import math
from .blocks import PoseDecoder, HeatmapDecoder, EfficientHeatmapDecoder

OptIntSeq = Optional[Sequence[int]]


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.kaiming_normal(m.weight)


class EnhancedEncoder(nn.Module):
	"""Encoder with configurable HMD info size."""
	def __init__(self, num_classes=16, output_size=64, hmd_info_size=9):
		super(EnhancedEncoder, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
		self.lrelu1 = nn.LeakyReLU(0.2)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.lrelu2 = nn.LeakyReLU(0.2)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.lrelu3 = nn.LeakyReLU(0.2)
		self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

		self.linear1 = nn.Linear(hmd_info_size, 36)
		self.lrelu4 = nn.LeakyReLU(0.2)

		self.linear2 = nn.Linear(256+36, output_size)
		self.linear2_ = nn.Linear(256, output_size)
		self.lrelu5 = nn.LeakyReLU(0.2)

	def forward(self, hm, hmd=None):
		hm = self.conv1(hm)
		hm = self.lrelu1(hm)
		hm = self.conv2(hm)
		hm = self.lrelu2(hm)
		hm = self.conv3(hm)
		hm = self.lrelu3(hm)

		hm_avgpool = self.avr_pool(hm).reshape(-1, 256)

		if hmd is not None:
			hmd = self.linear1(hmd)
			hmd = self.lrelu4(hmd)

			x = torch.cat((hm_avgpool, hmd), dim=1).to(torch.float32)

			x = self.linear2(x)
		else:
			x = self.linear2_(hm_avgpool)

		x = self.lrelu5(x)

		return x


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
				input_size=20,
				num_classes=16,
				linear_size=512,
				num_stage=1,
				p_dropout=0.5,
				):
		super(LinearModel, self).__init__()

		self.linear_size = linear_size
		self.p_dropout = p_dropout
		self.num_stage = num_stage

		# 2d joints
		self.input_size = input_size
		# 3d joints
		self.output_size = num_classes * 3

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
		y = y.reshape(-1, self.output_size // 3, 3)
		return y


@MODELS.register_module()
class CustomxRegoposeBaselinel1_enhanced_hmd(BaseHead):
	"""Baseline head with configurable HMD info size.

	This head is identical to CustomxRegoposeBaselinel1 but accepts a configurable
	hmd_info_size parameter to support enhanced HMD info from the EnhanceHMDInfo transform.

	Args:
		hmd_info_size (int): Size of HMD info input. Default: 9 (standard).
			Options: 10 (head_height), 11 (torso_reference), 12 (head_hands_y or relative_heights)
	"""

	_version = 2

	def __init__(self,
				in_channels: Union[int, Sequence[int]],
				out_channels: int,
				deconv_out_channels: OptIntSeq = (256, 256),
				deconv_kernel_sizes: OptIntSeq = (4, 4),
				deconv_stride_sizes: OptIntSeq = (2, 2),
				conv_out_channels: OptIntSeq = None,
				conv_kernel_sizes: OptIntSeq = None,
				final_layer: dict = dict(kernel_size=1),
				hmd_info_size: int = 9,  # NEW: configurable HMD info size
				loss: ConfigType = dict(
					type='KeypointMSELoss'),
				loss_pose_l2norm: ConfigType = dict(
					type='pose_l2norm'),
				loss_cosine_similarity: ConfigType = dict(
					type='cosine_similarity'),
				loss_limb_length: ConfigType = dict(
					type='limb_length'
				),
				loss_heatmap_recon: ConfigType = dict(
					type='KeypointMSELoss'
				),
				loss_hmd: ConfigType = dict(
					type='MSELoss'
				),
				loss_backbone_latant: ConfigType = dict(
					type='MSELoss'
				),
				loss_backbone_heatmap: ConfigType = dict(
					type='KeypointMSELoss'),
				loss_bone_length: OptConfigType = None,
				loss_symmetry: OptConfigType = None,
				decoder: OptConfigType = None,
				heatmap_decoder_type: str = 'original',
				init_cfg: OptConfigType = None):

		if init_cfg is None:
			init_cfg = self.default_init_cfg

		super().__init__(init_cfg)
		self.hm_iteration = 2000
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.hmd_info_size = hmd_info_size  # Store for reference

		self.loss_module = MODELS.build(loss)
		self.loss_pose_l2norm_module = MODELS.build(loss_pose_l2norm)
		self.loss_cosine_similarity_module = MODELS.build(loss_cosine_similarity)
		self.loss_limb_length_module = MODELS.build(loss_limb_length)
		self.loss_heatmap_recon_module = MODELS.build(loss_heatmap_recon)
		self.loss_hmd_module = MODELS.build(loss_hmd)

		# Optional structural losses
		if loss_bone_length is not None:
			self.loss_bone_length_module = MODELS.build(loss_bone_length)
		else:
			self.loss_bone_length_module = None

		if loss_symmetry is not None:
			self.loss_symmetry_module = MODELS.build(loss_symmetry)
		else:
			self.loss_symmetry_module = None

		# Use EnhancedEncoder with configurable hmd_info_size
		self.encoder = EnhancedEncoder(
			num_classes=out_channels,
			output_size=64,
			hmd_info_size=hmd_info_size
		)

		# Select heatmap decoder type
		self.heatmap_decoder_type = heatmap_decoder_type
		if heatmap_decoder_type == 'efficient':
			self.heatmap_decoder = EfficientHeatmapDecoder(
				num_classes=out_channels, heatmap_resolution=47, input_size=64)
		else:
			self.heatmap_decoder = HeatmapDecoder(
				num_classes=out_channels, heatmap_resolution=47, input_size=64)

		self.pose_decoder = LinearModel(
			input_size=64,
			num_classes=16,
			linear_size=512,
			num_stage=1,
			p_dropout=0.3
		)

		# HMD linear layer with configurable input size
		self.hmd_linear = nn.Sequential(
			nn.Linear(hmd_info_size, 64),
			nn.ReLU(inplace=True)
		)

		self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))
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
				layer_stride_sizes=deconv_stride_sizes,
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

		self.add_deconv_layers = nn.Sequential(
			nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True)
		)

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
							layer_kernel_sizes: Sequence[int],
							layer_stride_sizes: Sequence[int]) -> nn.Module:
		"""Create deconvolutional layers by given parameters."""

		layers = []
		for out_channels, kernel_size, stride_size in zip(layer_out_channels,
											 layer_kernel_sizes,
											 layer_stride_sizes):
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
				stride=stride_size,
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
		"""Forward the network."""
		x = self.deconv_layers(feats[-1])
		x = self.add_deconv_layers(x)
		x = self.conv_layers(x)
		x = self.final_layer(x)
		return x

	def decode(self, batch_outputs: Union[Tensor, Tuple[Tensor]], batch_data_samples: OptSampleList) -> InstanceList:
		"""Decode keypoints from outputs."""

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

		# HMD_info (now with configurable size)
		HMD_info = torch.cat([
			d.gt_instance_labels.hmd_info for d in batch_data_samples
		])

		z = self.encoder(batch_outputs.to(torch.float32))
		hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))

		batch_3d_keypoints = self.pose_decoder(z + hmd_info_)
		generated_heatmaps = self.heatmap_decoder(z + hmd_info_)

		def preprocess_hmd_data_batch(p3d):
			# Ensure p3d is a PyTorch tensor
			if not isinstance(p3d, torch.Tensor):
				p3d = torch.tensor(p3d, dtype=torch.float32)

			# Extract head and hand positions
			head = p3d[:, 0]
			right_hand = p3d[:, 7]
			left_hand = p3d[:, 4]

			# Create a local coordinate system
			midpoint = (right_hand + left_hand) / 2
			z_axis = midpoint - head
			z_axis = z_axis / torch.norm(z_axis, dim=1, keepdim=True)

			hand_vector = right_hand - left_hand
			x_axis = torch.cross(z_axis, hand_vector, dim=1)
			x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

			y_axis = torch.cross(z_axis, x_axis, dim=1)

			rotation_matrices = torch.stack((x_axis, y_axis, z_axis), dim=2)

			right_local = torch.bmm(rotation_matrices.transpose(1, 2), (right_hand - head).unsqueeze(2)).squeeze(2)
			left_local = torch.bmm(rotation_matrices.transpose(1, 2), (left_hand - head).unsqueeze(2)).squeeze(2)

			hand_distance = torch.norm(right_local - left_local, dim=1)
			right_distance = torch.norm(right_local, dim=1)
			left_distance = torch.norm(left_local, dim=1)

			# Return only base 9 dims for HMD reconstruction loss
			preprocessed_hmd = torch.cat([
				right_local, left_local,
				hand_distance.unsqueeze(1), right_distance.unsqueeze(1), left_distance.unsqueeze(1)
			], dim=1)

			return preprocessed_hmd

		hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)

		for keypoints, keypoint_3d, scores, visibility, generated_heatmap, hmd_recon in zip(
				batch_keypoints, batch_3d_keypoints, batch_scores,
				batch_visibility, generated_heatmaps, hmd_recons):
			keypoint_3d = keypoint_3d.unsqueeze(dim=0)
			hmd_recon = hmd_recon.unsqueeze(dim=0)
			generated_heatmap = generated_heatmap.unsqueeze(dim=0)
			pred = InstanceData(
				keypoints=keypoints,
				keypoint_scores=scores,
				keypoint_3d=keypoint_3d,
				generated_heatmap=generated_heatmap,
				hmd_recon=hmd_recon
			)
			if visibility is not None:
				pred.keypoints_visible = visibility
			preds.append(pred)

		return preds, batch_3d_keypoints

	def predict(self,
				feats: Features,
				batch_data_samples: OptSampleList,
				test_cfg: ConfigType = {}) -> Predictions:
		"""Predict results from features."""

		if test_cfg.get('flip_test', False):
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

		preds, _ = self.decode(batch_heatmaps, batch_data_samples)

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
		"""Calculate losses from a batch of inputs and data samples."""
		pred_fields = self.forward(feats)
		gt_heatmaps = torch.stack(
			[d.gt_fields.heatmaps for d in batch_data_samples])
		keypoint_weights = torch.cat([
			d.gt_instance_labels.keypoint_weights for d in batch_data_samples
		])

		pred, pred_batch_3d_keypoints = self.decode(pred_fields, batch_data_samples)

		pred_recon_heatmap = torch.cat([
			p.generated_heatmap for p in pred
		])

		pred_recon_hmd = torch.cat([
			p.hmd_recon for p in pred
		])

		gt_keypoint_3d = torch.cat([
			d.gt_instance_labels.keypoint3d for d in batch_data_samples
		])
		HMD_info = torch.cat([
			d.gt_instance_labels.hmd_info for d in batch_data_samples
		])

		# For HMD loss, use only first 9 dims (base HMD info)
		HMD_info_base = HMD_info[:, :9]

		pred_batch_3d_keypoints = pred_batch_3d_keypoints.reshape(-1, 16, 3)

		loss_pose_l2norm = self.loss_pose_l2norm_module(pred_batch_3d_keypoints, gt_keypoint_3d)
		loss_cosine_similarity = self.loss_cosine_similarity_module(pred_batch_3d_keypoints, gt_keypoint_3d)
		loss_limb_length = self.loss_limb_length_module(pred_batch_3d_keypoints, gt_keypoint_3d)
		loss_heatmap_recon = self.loss_heatmap_recon_module(pred_recon_heatmap, gt_heatmaps, keypoint_weights)
		loss_2dkpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
		loss_hmd = self.loss_hmd_module(pred_recon_hmd.to(torch.double), HMD_info_base.to(torch.double))

		losses = dict()

		losses.update(loss_pose_l2norm=torch.mean(loss_pose_l2norm))
		losses.update(loss_cosine_similarity=torch.mean(loss_cosine_similarity))
		losses.update(loss_limb_length=torch.mean(loss_limb_length))
		losses.update(loss_heatmap_recon=loss_heatmap_recon)
		losses.update(loss_hmd=loss_hmd)
		losses.update(loss_kpt=loss_2dkpt)

		# Optional structural losses
		if self.loss_bone_length_module is not None:
			loss_bone = self.loss_bone_length_module(
				pred_batch_3d_keypoints, gt_keypoint_3d)
			losses.update(loss_bone_length=torch.mean(loss_bone))

		if self.loss_symmetry_module is not None:
			loss_sym = self.loss_symmetry_module(pred_batch_3d_keypoints)
			losses.update(loss_symmetry=torch.mean(loss_sym))

		# calculate accuracy
		if train_cfg.get('compute_acc', True):
			_, avg_acc, _ = pose_pck_accuracy(
				output=to_numpy(pred_fields),
				target=to_numpy(gt_heatmaps),
				mask=to_numpy(keypoint_weights) > 0)

			acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
			losses.update(acc_pose=acc_pose)

		self.hm_iteration += 1

		return losses

	def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
								  **kwargs):
		"""Hook to convert old-version state dict."""
		version = local_meta.get('version', None)
		if version and version >= self._version:
			return

		keys = list(state_dict.keys())
		for _k in keys:
			if not _k.startswith(prefix):
				continue
			v = state_dict.pop(_k)
			k = _k[len(prefix):]
			k_parts = k.split('.')
			if k_parts[0] == 'final_layer':
				if len(k_parts) == 3:
					assert isinstance(self.conv_layers, nn.Sequential)
					idx = int(k_parts[1])
					if idx < len(self.conv_layers):
						k_new = 'conv_layers.' + '.'.join(k_parts[1:])
					else:
						k_new = 'final_layer.' + k_parts[2]
				else:
					k_new = k
			else:
				k_new = k

			state_dict[prefix + k_new] = v
