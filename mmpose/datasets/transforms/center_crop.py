# Copyright (c) OpenMMLab. All rights reserved.
"""
Center Crop Transform for Mo2Cap2 Test Images

Crops horizontal margins from fisheye test images to remove vignetting.
Mo2Cap2 test images: 1280x1024, crop 128px from each side -> 1024x1024.
"""
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Mo2Cap2CenterCrop(BaseTransform):
    """Crop horizontal margins from Mo2Cap2 test images.

    Mo2Cap2 test images are 1280x1024 with vignetting at edges.
    This transform crops 128px from each horizontal side -> 1024x1024.

    Args:
        margin_left (int): Pixels to crop from left. Default: 128
        margin_right (int): Pixels to crop from right. Default: 128
        margin_top (int): Pixels to crop from top. Default: 0
        margin_bottom (int): Pixels to crop from bottom. Default: 0
    """

    def __init__(self,
                 margin_left: int = 128,
                 margin_right: int = 128,
                 margin_top: int = 0,
                 margin_bottom: int = 0):
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom

    def transform(self, results: dict) -> dict:
        """Crop margins from image.

        Args:
            results: Dict containing 'img' key with image array.

        Returns:
            Modified results dict with cropped image.
        """
        img = results['img']
        h, w = img.shape[:2]

        # Calculate crop region
        x1 = self.margin_left
        x2 = w - self.margin_right
        y1 = self.margin_top
        y2 = h - self.margin_bottom

        # Crop
        img_cropped = img[y1:y2, x1:x2].copy()
        results['img'] = img_cropped

        # New dimensions
        new_h, new_w = img_cropped.shape[:2]

        # Update image shape info if present
        if 'img_shape' in results:
            results['img_shape'] = (new_h, new_w)
        if 'ori_shape' in results:
            results['ori_shape'] = (new_h, new_w)

        # Update bbox to reflect new cropped image dimensions
        # Dataset sets bbox assuming default crop, we need to update it
        if 'bbox' in results:
            # bbox is (N, 4) in xyxy format, update to cover full cropped image
            results['bbox'] = np.array([[0, 0, new_w, new_h]], dtype=np.float32)

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'margin_left={self.margin_left}, '
                f'margin_right={self.margin_right}, '
                f'margin_top={self.margin_top}, '
                f'margin_bottom={self.margin_bottom})')
