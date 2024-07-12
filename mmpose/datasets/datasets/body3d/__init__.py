# Copyright (c) OpenMMLab. All rights reserved.
from .h36m_dataset import Human36mDataset
from .custom_egopose_dataset import CustomEgoposeDataset
from .unity_36m_dataset import Unity36mDataset
__all__ = [
	'Human36mDataset',
	'CustomEgoposeDataset',
	'Unity36mDataset'
	]
