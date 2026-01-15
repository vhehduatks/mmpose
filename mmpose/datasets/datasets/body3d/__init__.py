# Copyright (c) OpenMMLab. All rights reserved.
from .h36m_dataset import Human36mDataset
from .custom_egopose_dataset import CustomEgoposeDataset
from .unity_36m_dataset import Unity36mDataset
from .custom_egopose_dataset_seg_depth import CustomEgoposeDataset_seg_depth
from .custom_egopose_dataset_h5cache import (
	H5CachedEgoposeDataset,
	H5CachedEgoposeDataset_SegDepth
)
from .h5_mo2cap2_dataset import (
	H5Mo2Cap2Dataset,
	H5Mo2Cap2Dataset_Lazy
)

__all__ = [
	'Human36mDataset',
	'CustomEgoposeDataset',
	'Unity36mDataset',
	'CustomEgoposeDataset_seg_depth',
	'H5CachedEgoposeDataset',
	'H5CachedEgoposeDataset_SegDepth',
	'H5Mo2Cap2Dataset',
	'H5Mo2Cap2Dataset_Lazy'
	]
