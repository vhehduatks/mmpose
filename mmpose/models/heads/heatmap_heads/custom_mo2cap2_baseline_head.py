# Copyright (c) OpenMMLab. All rights reserved.
"""
Mo2Cap2 Baseline Head - Simple Heatmap to 3D Lifting

This is a simple baseline without cascaded refinement:
1. Backbone -> Deconv -> Heatmap (47x47)
2. Encoder: Heatmap + HMD info -> Latent Z (64-dim)
3. Pose Decoder: Z -> 3D Pose (15 joints)
4. Heatmap Decoder: Z -> Reconstructed Heatmap

Mo2Cap2 Joint Indices (15 joints):
  - 0: Neck (root)
  - 1-3: RightArm, RightForeArm, RightHand
  - 4-6: LeftArm, LeftForeArm, LeftHand
  - 7-10: RightUpLeg, RightLeg, RightFoot, RightToeBase
  - 11-14: LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase

Reference:
  "A simple yet effective baseline for 3d human pose estimation"
  Martinez et al., ICCV 2017
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
from .blocks import HeatmapDecoder, EfficientHeatmapDecoder

OptIntSeq = Optional[Sequence[int]]


class Encoder(nn.Module):
    """Encoder: Heatmap + HMD info -> Latent Z."""

    def __init__(self, num_classes=15, output_size=64, hmd_info_size=9):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.Linear(hmd_info_size, 36)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(256 + 36, output_size)
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
    """Residual linear block."""

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
        return x + y


class LinearModel(nn.Module):
    """Linear model for 2D-to-3D lifting."""

    def __init__(self, input_size=64, num_classes=15, linear_size=512,
                 num_stage=1, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = input_size
        self.output_size = num_classes * 3

        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = nn.ModuleList([
            Linear(self.linear_size, self.p_dropout) for _ in range(num_stage)
        ])

        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.linear_stages:
            y = stage(y)
        y = self.w2(y)
        y = y.reshape(-1, self.output_size // 3, 3)
        return y


def preprocess_hmd_data_batch_mo2cap2(p3d):
    """Compute HMD reconstruction from predicted 3D pose for Mo2Cap2.

    Mo2Cap2 Joint Indices:
      - 0: Neck (root/head equivalent)
      - 3: RightHand
      - 6: LeftHand
    """
    if not isinstance(p3d, torch.Tensor):
        p3d = torch.tensor(p3d, dtype=torch.float32)

    head = p3d[:, 0]        # Neck
    right_hand = p3d[:, 3]  # RightHand
    left_hand = p3d[:, 6]   # LeftHand

    midpoint = (right_hand + left_hand) / 2
    z_axis = midpoint - head
    z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-8)

    hand_vector = right_hand - left_hand
    x_axis = torch.cross(z_axis, hand_vector, dim=1)
    x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-8)

    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rotation_matrices = torch.stack((x_axis, y_axis, z_axis), dim=2)

    right_local = torch.bmm(
        rotation_matrices.transpose(1, 2),
        (right_hand - head).unsqueeze(2)
    ).squeeze(2)
    left_local = torch.bmm(
        rotation_matrices.transpose(1, 2),
        (left_hand - head).unsqueeze(2)
    ).squeeze(2)

    hand_distance = torch.norm(right_local - left_local, dim=1)
    right_distance = torch.norm(right_local, dim=1)
    left_distance = torch.norm(left_local, dim=1)

    return torch.cat([
        right_local, left_local,
        hand_distance.unsqueeze(1),
        right_distance.unsqueeze(1),
        left_distance.unsqueeze(1)
    ], dim=1)


@MODELS.register_module()
class CustomMo2Cap2BaselineHead(BaseHead):
    """Mo2Cap2 Baseline Head - Simple Heatmap to 3D Lifting.

    Architecture:
        Backbone -> Deconv -> Heatmap (47x47)
                           -> Encoder (with HMD) -> Z (64-dim)
                           -> Pose Decoder -> 3D Pose (15 joints)
                           -> Heatmap Decoder -> Recon Heatmap

    No cascaded refinement - direct lifting from heatmap features.

    Args:
        in_channels (int): Input channels from backbone.
        out_channels (int): Number of output joints (15 for Mo2Cap2).
        hmd_info_size (int): Size of HMD info. Default: 9.
        heatmap_decoder_type (str): 'original' or 'efficient'. Default: 'efficient'.
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int = 15,
                 deconv_out_channels: OptIntSeq = (256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4),
                 deconv_stride_sizes: OptIntSeq = (2, 2),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 hmd_info_size: int = 9,
                 loss: ConfigType = dict(type='KeypointMSELoss'),
                 loss_pose_l2norm: ConfigType = dict(type='pose_l2norm'),
                 loss_cosine_similarity: ConfigType = dict(type='cosine_similarity'),
                 loss_limb_length: ConfigType = dict(type='limb_length'),
                 loss_heatmap_recon: ConfigType = dict(type='KeypointMSELoss'),
                 loss_hmd: ConfigType = dict(type='MSELoss'),
                 loss_bone_length: OptConfigType = None,
                 loss_symmetry: OptConfigType = None,
                 decoder: OptConfigType = None,
                 heatmap_decoder_type: str = 'efficient',
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hmd_info_size = hmd_info_size

        # Loss modules
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

        # Encoder
        self.encoder = Encoder(
            num_classes=out_channels,
            output_size=64,
            hmd_info_size=hmd_info_size
        )

        # Heatmap decoder
        self.heatmap_decoder_type = heatmap_decoder_type
        if heatmap_decoder_type == 'efficient':
            self.heatmap_decoder = EfficientHeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)
        else:
            self.heatmap_decoder = HeatmapDecoder(
                num_classes=out_channels, heatmap_resolution=47, input_size=64)

        # Pose decoder (15 joints for Mo2Cap2)
        self.pose_decoder = LinearModel(
            input_size=64,
            num_classes=out_channels,
            linear_size=512,
            num_stage=1,
            p_dropout=0.3
        )

        # HMD embedding
        self.hmd_linear = nn.Sequential(
            nn.Linear(hmd_info_size, 64),
            nn.ReLU(inplace=True)
        )

        self.avr_pool = nn.AdaptiveAvgPool2d((1, 1))

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Deconv layers
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length.')
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
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length.')
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

        # Upsample to 47x47
        self.add_deconv_layers = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels, layer_out_channels, layer_kernel_sizes):
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(type='Conv2d', in_channels=in_channels,
                       out_channels=out_channels, kernel_size=kernel_size,
                       stride=1, padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels, layer_out_channels,
                            layer_kernel_sizes, layer_stride_sizes):
        layers = []
        for out_channels, kernel_size, stride_size in zip(
                layer_out_channels, layer_kernel_sizes, layer_stride_sizes):
            if kernel_size == 4:
                padding, output_padding = 1, 0
            elif kernel_size == 3:
                padding, output_padding = 1, 1
            elif kernel_size == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size}')
            cfg = dict(type='deconv', in_channels=in_channels,
                       out_channels=out_channels, kernel_size=kernel_size,
                       stride=stride_size, padding=padding,
                       output_padding=output_padding, bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def decode(self, batch_outputs, batch_data_samples):
        """Decode keypoints from heatmap outputs."""

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}.')

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
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        # HMD info
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        # Encode + predict
        z = self.encoder(batch_outputs.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = z + hmd_info_

        batch_3d_keypoints = self.pose_decoder(z_plus_hmd)
        generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
        hmd_recons = preprocess_hmd_data_batch_mo2cap2(batch_3d_keypoints)

        # Pack results
        preds = []
        for (keypoints, kp3d, scores, visibility, gen_hm, hmd_rec) in zip(
                batch_keypoints, batch_3d_keypoints, batch_scores,
                batch_visibility, generated_heatmaps, hmd_recons):
            kp3d = kp3d.unsqueeze(dim=0)
            hmd_rec = hmd_rec.unsqueeze(dim=0)
            gen_hm = gen_hm.unsqueeze(dim=0)
            pred = InstanceData(
                keypoints=keypoints, keypoint_scores=scores,
                keypoint_3d=kp3d, generated_heatmap=gen_hm,
                hmd_recon=hmd_rec)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, batch_3d_keypoints

    def predict(self, feats, batch_data_samples, test_cfg={}):
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

    def loss(self, feats, batch_data_samples, train_cfg={}):
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples
        ])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples
        ])

        pred, pred_batch_3d_keypoints = self.decode(pred_fields, batch_data_samples)

        pred_recon_heatmap = torch.cat([p.generated_heatmap for p in pred])
        pred_recon_hmd = torch.cat([p.hmd_recon for p in pred])

        pred_batch_3d_keypoints = pred_batch_3d_keypoints.reshape(-1, self.out_channels, 3)

        # Losses
        loss_pose_l2norm = self.loss_pose_l2norm_module(
            pred_batch_3d_keypoints, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            pred_batch_3d_keypoints, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(
            pred_batch_3d_keypoints, gt_keypoint_3d)
        loss_heatmap_recon = self.loss_heatmap_recon_module(
            pred_recon_heatmap, gt_heatmaps, keypoint_weights)
        loss_kpt = self.loss_module(
            pred_fields, gt_heatmaps, keypoint_weights)

        # HMD loss uses base 9 dims only
        HMD_info_base = HMD_info[:, :9]
        loss_hmd = self.loss_hmd_module(
            pred_recon_hmd.to(torch.double), HMD_info_base.to(torch.double))

        losses = dict()
        losses.update(loss_pose_l2norm=torch.mean(loss_pose_l2norm))
        losses.update(loss_cosine_similarity=torch.mean(loss_cosine))
        losses.update(loss_limb_length=torch.mean(loss_limb))
        losses.update(loss_heatmap_recon=loss_heatmap_recon)
        losses.update(loss_hmd=loss_hmd)
        losses.update(loss_kpt=loss_kpt)

        # Optional structural losses
        if self.loss_bone_length_module is not None:
            loss_bone = self.loss_bone_length_module(
                pred_batch_3d_keypoints, gt_keypoint_3d)
            losses.update(loss_bone_length=torch.mean(loss_bone))

        if self.loss_symmetry_module is not None:
            loss_sym = self.loss_symmetry_module(pred_batch_3d_keypoints)
            losses.update(loss_symmetry=torch.mean(loss_sym))

        # Accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta,
                                  *args, **kwargs):
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
