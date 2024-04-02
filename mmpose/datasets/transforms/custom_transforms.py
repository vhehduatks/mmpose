import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.image import imflip
from mmcv.transforms import BaseTransform


from mmpose.codecs import *  # noqa: F401, F403
from mmpose.registry import KEYPOINT_CODECS, TRANSFORMS


try:
    import albumentations
except ImportError:
    albumentations = None

Number = Union[int, float]

@TRANSFORMS.register_module()
class EgoposeFilterAnnotations(BaseTransform):
    """Eliminate undesirable annotations based on specific conditions.

    This class is designed to sift through annotations by examining multiple
    factors such as the size of the bounding box, the visibility of keypoints,
    and the overall area. Users can fine-tune the criteria to filter out
    instances that have excessively small bounding boxes, insufficient area,
    or an inadequate number of visible keypoints.

    Required Keys:

    - bbox (np.ndarray) (optional)
    - area (np.int64) (optional)
    - keypoints_visible (np.ndarray) (optional)

    Modified Keys:

    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground
            truth boxes. Default: (1., 1.)
        min_gt_area (int): Minimum foreground area of instances.
            Default: 1
        min_kpt_vis (int): Minimum number of visible keypoints. Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: False
        by_area (bool): Filter instances with area less than min_gt_area
            threshold. Default: False
        by_kpt (bool): Filter instances with keypoints_visible not meeting the
            min_kpt_vis threshold. Default: True
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_area: int = 1,
                 min_kpt_vis: int = 1,
                 by_box: bool = False,
                 by_area: bool = False,
                 by_kpt: bool = True,
                 keep_empty: bool = True) -> None:

        assert by_box or by_kpt or by_area
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_area = min_gt_area
        self.min_kpt_vis = min_kpt_vis
        self.by_box = by_box
        self.by_area = by_area
        self.by_kpt = by_kpt
        self.keep_empty = keep_empty

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'keypoints' in results
        kpts = results['keypoints']
        if kpts.shape[0] == 0:
            return results

        tests = []
        if self.by_box and 'bbox' in results:
            # bbox = results['bbox']
            # tests.append(
            #     ((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) &
            #      (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
            pass
        if self.by_area and 'area' in results:
            # area = results['area']
            # tests.append(area >= self.min_gt_area)
            pass
        if self.by_kpt:
            tests.append(len(kpts) == 16)

		## temp code 
        # tests.append(True)
		##
        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep:
            if self.keep_empty:
                return None

        # keys = ('bbox', 'bbox_score', 'category_id', 'keypoints',
        #         'keypoints_visible', 'area')
        
        # for key in keys:
        #     if key in results:
        #         results[key] = results[key][keep]

        return results

    def __repr__(self):
        # return (f'{self.__class__.__name__}('
        #         f'min_gt_bbox_wh={self.min_gt_bbox_wh}, '
        #         f'min_gt_area={self.min_gt_area}, '
        #         f'min_kpt_vis={self.min_kpt_vis}, '
        #         f'by_box={self.by_box}, '
        #         f'by_area={self.by_area}, '
        #         f'by_kpt={self.by_kpt}, '
        #         f'keep_empty={self.keep_empty})')
        pass


# @TRANSFORMS.register_module()
# class GenerateTarget(BaseTransform):
#     """Encode keypoints into Target.

#     The generated target is usually the supervision signal of the model
#     learning, e.g. heatmaps or regression labels.

#     Required Keys:

#         - keypoints
#         - keypoints_visible
#         - dataset_keypoint_weights

#     Added Keys:

#         - The keys of the encoded items from the codec will be updated into
#             the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
#             the specific codec for more details.

#     Args:
#         encoder (dict | list[dict]): The codec config for keypoint encoding.
#             Both single encoder and multiple encoders (given as a list) are
#             supported
#         multilevel (bool): Determine the method to handle multiple encoders.
#             If ``multilevel==True``, generate multilevel targets from a group
#             of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
#             encoders with different sigma values); If ``multilevel==False``,
#             generate combined targets from a group of different encoders. This
#             argument will have no effect in case of single encoder. Defaults
#             to ``False``
#         use_dataset_keypoint_weights (bool): Whether use the keypoint weights
#             from the dataset meta information. Defaults to ``False``
#         target_type (str, deprecated): This argument is deprecated and has no
#             effect. Defaults to ``None``
#     """

#     def __init__(self,
#                  encoder: MultiConfig,
#                  target_type: Optional[str] = None,
#                  multilevel: bool = False,
#                  use_dataset_keypoint_weights: bool = False) -> None:
#         super().__init__()

#         if target_type is not None:
#             rank, _ = get_dist_info()
#             if rank == 0:
#                 warnings.warn(
#                     'The argument `target_type` is deprecated in'
#                     ' GenerateTarget. The target type and encoded '
#                     'keys will be determined by encoder(s).',
#                     DeprecationWarning)

#         self.encoder_cfg = deepcopy(encoder)
#         self.multilevel = multilevel
#         self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

#         if isinstance(self.encoder_cfg, list):
#             self.encoder = [
#                 KEYPOINT_CODECS.build(cfg) for cfg in self.encoder_cfg
#             ]
#         else:
#             assert not self.multilevel, (
#                 'Need multiple encoder configs if ``multilevel==True``')
#             self.encoder = KEYPOINT_CODECS.build(self.encoder_cfg)

#     def transform(self, results: Dict) -> Optional[dict]:
#         """The transform function of :class:`GenerateTarget`.

#         See ``transform()`` method of :class:`BaseTransform` for details.
#         """

#         if results.get('transformed_keypoints', None) is not None:
#             # use keypoints transformed by TopdownAffine
#             keypoints = results['transformed_keypoints']
#         elif results.get('keypoints', None) is not None:
#             # use original keypoints
#             keypoints = results['keypoints']
#         else:
#             raise ValueError(
#                 'GenerateTarget requires \'transformed_keypoints\' or'
#                 ' \'keypoints\' in the results.')

#         keypoints_visible = results['keypoints_visible']
#         if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
#             keypoints_visible, keypoints_visible_weights = \
#                 keypoints_visible[..., 0], keypoints_visible[..., 1]
#             results['keypoints_visible'] = keypoints_visible
#             results['keypoints_visible_weights'] = keypoints_visible_weights

#         # Encoded items from the encoder(s) will be updated into the results.
#         # Please refer to the document of the specific codec for details about
#         # encoded items.
#         if not isinstance(self.encoder, list):
#             # For single encoding, the encoded items will be directly added
#             # into results.
#             auxiliary_encode_kwargs = {
#                 key: results[key]
#                 for key in self.encoder.auxiliary_encode_keys
#             }
#             encoded = self.encoder.encode(
#                 keypoints=keypoints,
#                 keypoints_visible=keypoints_visible,
#                 **auxiliary_encode_kwargs)

#             if self.encoder.field_mapping_table:
#                 encoded[
#                     'field_mapping_table'] = self.encoder.field_mapping_table
#             if self.encoder.instance_mapping_table:
#                 encoded['instance_mapping_table'] = \
#                     self.encoder.instance_mapping_table
#             if self.encoder.label_mapping_table:
#                 encoded[
#                     'label_mapping_table'] = self.encoder.label_mapping_table

#         else:
#             encoded_list = []
#             _field_mapping_table = dict()
#             _instance_mapping_table = dict()
#             _label_mapping_table = dict()
#             for _encoder in self.encoder:
#                 auxiliary_encode_kwargs = {
#                     key: results[key]
#                     for key in _encoder.auxiliary_encode_keys
#                 }
#                 encoded_list.append(
#                     _encoder.encode(
#                         keypoints=keypoints,
#                         keypoints_visible=keypoints_visible,
#                         **auxiliary_encode_kwargs))

#                 _field_mapping_table.update(_encoder.field_mapping_table)
#                 _instance_mapping_table.update(_encoder.instance_mapping_table)
#                 _label_mapping_table.update(_encoder.label_mapping_table)

#             if self.multilevel:
#                 # For multilevel encoding, the encoded items from each encoder
#                 # should have the same keys.

#                 keys = encoded_list[0].keys()
#                 if not all(_encoded.keys() == keys
#                            for _encoded in encoded_list):
#                     raise ValueError(
#                         'Encoded items from all encoders must have the same '
#                         'keys if ``multilevel==True``.')

#                 encoded = {
#                     k: [_encoded[k] for _encoded in encoded_list]
#                     for k in keys
#                 }

#             else:
#                 # For combined encoding, the encoded items from different
#                 # encoders should have no overlapping items, except for
#                 # `keypoint_weights`. If multiple `keypoint_weights` are given,
#                 # they will be multiplied as the final `keypoint_weights`.

#                 encoded = dict()
#                 keypoint_weights = []

#                 for _encoded in encoded_list:
#                     for key, value in _encoded.items():
#                         if key == 'keypoint_weights':
#                             keypoint_weights.append(value)
#                         elif key not in encoded:
#                             encoded[key] = value
#                         else:
#                             raise ValueError(
#                                 f'Overlapping item "{key}" from multiple '
#                                 'encoders, which is not supported when '
#                                 '``multilevel==False``')

#                 if keypoint_weights:
#                     encoded['keypoint_weights'] = keypoint_weights

#             if _field_mapping_table:
#                 encoded['field_mapping_table'] = _field_mapping_table
#             if _instance_mapping_table:
#                 encoded['instance_mapping_table'] = _instance_mapping_table
#             if _label_mapping_table:
#                 encoded['label_mapping_table'] = _label_mapping_table

#         if self.use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
#             if isinstance(encoded['keypoint_weights'], list):
#                 for w in encoded['keypoint_weights']:
#                     w = w * results['dataset_keypoint_weights']
#             else:
#                 encoded['keypoint_weights'] = encoded[
#                     'keypoint_weights'] * results['dataset_keypoint_weights']

#         results.update(encoded)

#         return results

#     def __repr__(self) -> str:
#         """print the basic information of the transform.

#         Returns:
#             str: Formatted string.
#         """
#         repr_str = self.__class__.__name__
#         repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
#         repr_str += ('use_dataset_keypoint_weights='
#                      f'{self.use_dataset_keypoint_weights})')
#         return repr_str