from .base import BaseKeypointCodec
from typing import Any, List, Optional, Tuple
import numpy as np
from mmpose.registry import KEYPOINT_CODECS
import cv2

@KEYPOINT_CODECS.register_module()
class Egoposecodec(BaseKeypointCodec):

	auxiliary_encode_keys = {'depth_path', 'keypoints_3d'}
	field_mapping_table = {'depthmap':'depthmap'}
	instance_mapping_table = {'keypoints':'keypoints', 'keypoints_3d':'lifting_target', 'keypoints_visible':'lifting_target_visible' }
	label_mapping_table = {'keypoints':'keypoints', 'keypoints_3d':'lifting_target', 'keypoints_visible':'lifting_target_visible'}

	
	def __init__(self,
					expand_bbox: bool = False,
					input_size: Optional[Tuple] = None):
		super().__init__()
	def encode(self,
			depth_path:str,
			keypoints: np.ndarray,
			keypoints_3d:np.ndarray,
			keypoints_visible: Optional[np.ndarray] = None,
			) -> dict:
		"""Encode keypoints.

		Note:

			- instance number: N
			- keypoint number: K
			- keypoint dimension: D

		Args: 
			keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
			keypoints_visible (np.ndarray): Keypoint visibility in shape
				(N, K, D)

		Returns:
			dict: Encoded items.
		"""

		depthmap = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
		h, w = depthmap.shape[:2]
		
		# Find non-zero pixel range
		# Find non-zero pixel range
		nonzero_mask = depthmap > 0
		nonzero_rows = np.any(nonzero_mask, axis=1)
		nonzero_cols = np.any(nonzero_mask, axis=0)
		ymin, ymax = np.where(nonzero_rows)[0][[0, -1]]
		xmin, xmax = np.where(nonzero_cols)[0][[0, -1]]

		# Crop image
		cropped_depthmap = depthmap[ymin:ymax+1, xmin:xmax+1]

		encoded = {}
		encoded['depthmap'] = cropped_depthmap
		encoded['lifting_target'] = keypoints_3d
		encoded['lifting_target_visible'] = keypoints_visible

		# encoded['keypoints_visible'] = keypoints_visible
		
		return encoded
	
	def decode(self, encoded: Any) -> Tuple[np.ndarray]:
		pass

