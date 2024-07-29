# Copyright (c) OpenMMLab. All rights reserved.
from .annotation_processors import YOLOXPoseAnnotationProcessor
from .associative_embedding import AssociativeEmbedding
from .decoupled_heatmap import DecoupledHeatmap
from .edpose_label import EDPoseLabel
from .hand_3d_heatmap import Hand3DHeatmap
from .image_pose_lifting import ImagePoseLifting
from .integral_regression_label import IntegralRegressionLabel
from .megvii_heatmap import MegviiHeatmap
from .motionbert_label import MotionBERTLabel
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .spr import SPR
from .udp_heatmap import UDPHeatmap
from .video_pose_lifting import VideoPoseLifting
from .custom_codecs import Egoposecodec

from .custom_msra_heatmap import Custom_MSRAHeatmap
from .custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap

__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'IntegralRegressionLabel', 'AssociativeEmbedding', 'SPR',
    'DecoupledHeatmap', 'VideoPoseLifting', 'ImagePoseLifting',
    'MotionBERTLabel', 'YOLOXPoseAnnotationProcessor', 'EDPoseLabel',
    'Hand3DHeatmap', 'Egoposecodec', 'Custom_MSRAHeatmap', 'Custom_mo2cap2_MSRAHeatmap'
]
