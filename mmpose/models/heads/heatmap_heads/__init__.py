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

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead', 'InternetHead',
	'CustomHeatmapHead', 'CustomMo2Cap2HeatmapHead'
]
