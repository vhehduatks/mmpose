# Copyright (c) OpenMMLab. All rights reserved.
"""Direct full-image resize for egocentric pose estimation.

Replaces GetBBoxCenterScale + TopdownAffine for egocentric datasets where the
entire image should be resized to the model input size without any
bbox-based cropping.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EgoImageResize(BaseTransform):
    """Resize the full egocentric image to model input size.

    Unlike GetBBoxCenterScale + TopdownAffine, which crops a square bbox region
    and resizes it, this transform directly resizes the raw image (e.g.
    1920x1080 -> 256x256) so the model sees the entire field of view.

    Required Keys:
        - img
        - keypoints (optional)

    Modified Keys:
        - img

    Added Keys:
        - input_size
        - input_center
        - input_scale
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): Target (w, h). Default: (256, 256).
    """

    def __init__(self, input_size: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.input_size = input_size          # (w, h)

    def transform(self, results: Dict) -> Optional[Dict]:
        img = results['img']
        src_h, src_w = img.shape[:2]
        dst_w, dst_h = self.input_size

        # Simple resize (stretches to target size)
        results['img'] = cv2.resize(
            img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        # Scale factors for keypoint coordinate transform
        sx = dst_w / src_w
        sy = dst_h / src_h

        # Transform 2D keypoints to resized coordinate space
        if results.get('keypoints', None) is not None:
            kpts = results['keypoints'].copy()
            kpts[..., 0] *= sx
            kpts[..., 1] *= sy
            results['transformed_keypoints'] = kpts

        # Metadata expected by downstream components (codec, heads, etc.)
        results['input_size'] = (dst_w, dst_h)
        results['input_center'] = np.array(
            [src_w / 2.0, src_h / 2.0], dtype=np.float32)
        results['input_scale'] = np.array(
            [src_w, src_h], dtype=np.float32)

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_size={self.input_size})'
