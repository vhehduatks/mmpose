# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .internet_head import InternetHead
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .custom_heatmap_head import CustomHeatmapHead
from .custom_mo2cap2_heatmap_head import CustomMo2Cap2HeatmapHead
from .custom_mo2cap2_baselinel1_head import CustomMo2Cap2Baselinel1
from .custom_mo2cap2_baselinel1_head_multi_backbone import CustomMo2Cap2Baselinel1_multi_backbone
from .custom_egopose_baselinel1_head import CustomxRegoposeBaselinel1
from .custom_egopose_baselinel1_head_multi_backbone import CustomxRegoposeBaselinel1_multi_backbone
from .custom_egopose_baselinel1_head_seg_depth import CustomxRegoposeBaselinel1_multi_backbone_segdepth
from .custom_egopose_confidence_weighted_head import ConfidenceWeightedHMDHead

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead', 'InternetHead',
	'CustomHeatmapHead', 'CustomMo2Cap2HeatmapHead', 'CustomMo2Cap2Baselinel1', 'CustomMo2Cap2Baselinel1_multi_backbone',
	'CustomxRegoposeBaselinel1_multi_backbone', 'CustomxRegoposeBaselinel1',
	'CustomxRegoposeBaselinel1_multi_backbone_segdepth', 'ConfidenceWeightedHMDHead'
]
