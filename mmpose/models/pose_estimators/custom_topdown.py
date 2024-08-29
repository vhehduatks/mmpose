# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional
from typing import Tuple, Union

import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
from .topdown import TopdownPoseEstimator

@MODELS.register_module()
class Custom_TopdownPoseEstimator(TopdownPoseEstimator):
    def __init__(self,
                 backbone: ConfigType,
                 backbone2: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super(Custom_TopdownPoseEstimator, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        self.backbone2 = MODELS.build(backbone2)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
                resolutions.
        """
        x1 = self.backbone(inputs)
        x2 = self.backbone2(inputs)
        x = (x1,x2)
        if self.with_neck:
            x = self.neck(x)
        return x
