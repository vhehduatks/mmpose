# Copyright (c) OpenMMLab. All rights reserved.
from .coco_metric import CocoMetric
from .coco_wholebody_metric import CocoWholeBodyMetric
from .hand_metric import InterHandMetric
from .keypoint_2d_metrics import (AUC, EPE, NME, JhmdbPCKAccuracy,
                                  MpiiPCKAccuracy, PCKAccuracy)
from .keypoint_3d_metrics import MPJPE
from .keypoint_partition_metric import KeypointPartitionMetric
from .posetrack18_metric import PoseTrack18Metric
from .simple_keypoint_3d_metrics import SimpleMPJPE

from .custom_coco_metric import CustomCocoMetric
from .custom_keypoint_3d_metrics import Custom_MPJPE
from .custom_mo2cap2_metric import CustomMo2Cap2Metric
from .custom_xr_egopose_metric import CustomxRegoposeMetric

__all__ = [
    'CocoMetric', 'PCKAccuracy', 'MpiiPCKAccuracy', 'JhmdbPCKAccuracy', 'AUC',
    'EPE', 'NME', 'PoseTrack18Metric', 'CocoWholeBodyMetric',
    'KeypointPartitionMetric', 'MPJPE', 'InterHandMetric', 'SimpleMPJPE','CustomCocoMetric', 
	'Custom_MPJPE', 'CustomMo2Cap2Metric', 'CustomxRegoposeMetric'
]
