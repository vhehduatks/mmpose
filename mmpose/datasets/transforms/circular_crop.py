# Copyright (c) OpenMMLab. All rights reserved.
"""
Circular Crop Transform for Fisheye Images

Removes vignetting from fisheye images by applying a circular mask
that keeps the center region and fades/removes the dark corners.
"""
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CircularCrop(BaseTransform):
    """Apply circular crop/mask to remove vignetting from fisheye images.

    This transform creates a circular mask centered on the image and either:
    1. Fades the outer region smoothly (soft mask)
    2. Sets outer region to a fill color (hard mask)

    Args:
        radius_ratio (float): Ratio of circle radius to half the image size.
            1.0 means the circle touches the image edges.
            0.85 means 85% of the radius, removing corners.
            Default: 0.85
        soft_edge (bool): If True, apply smooth fade at edges.
            If False, hard cutoff. Default: True
        edge_width (float): Width of the soft edge as ratio of radius.
            Only used if soft_edge=True. Default: 0.1
        fill_value (int or tuple): Value to fill outside the circle.
            Default: 128 (gray)
    """

    def __init__(self,
                 radius_ratio: float = 0.85,
                 soft_edge: bool = True,
                 edge_width: float = 0.1,
                 fill_value: int = 128):
        self.radius_ratio = radius_ratio
        self.soft_edge = soft_edge
        self.edge_width = edge_width
        self.fill_value = fill_value
        self._mask_cache = {}

    def _get_circular_mask(self, h: int, w: int) -> np.ndarray:
        """Generate circular mask for given image size.

        Args:
            h: Image height
            w: Image width

        Returns:
            Mask array of shape (h, w) with values in [0, 1]
        """
        cache_key = (h, w, self.radius_ratio, self.soft_edge, self.edge_width)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Create coordinate grid
        center_y, center_x = h / 2, w / 2
        y, x = np.ogrid[:h, :w]

        # Distance from center (normalized)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = min(center_x, center_y)  # Use smaller dimension

        # Normalize distance
        dist_normalized = dist / max_dist

        # Create mask
        radius = self.radius_ratio

        if self.soft_edge:
            # Smooth transition at edges
            inner_radius = radius * (1 - self.edge_width)
            mask = np.clip((radius - dist_normalized) / (radius - inner_radius), 0, 1)
        else:
            # Hard cutoff
            mask = (dist_normalized <= radius).astype(np.float32)

        # Cache for reuse
        self._mask_cache[cache_key] = mask
        return mask

    def transform(self, results: dict) -> dict:
        """Apply circular mask to image.

        Args:
            results: Dict containing 'img' key with image array.

        Returns:
            Modified results dict with masked image.
        """
        img = results['img']
        h, w = img.shape[:2]

        # Get mask
        mask = self._get_circular_mask(h, w)

        # Apply mask
        if len(img.shape) == 3:
            # Color image: expand mask to 3 channels
            mask_3d = mask[:, :, np.newaxis]

            # Blend with fill value
            if isinstance(self.fill_value, (list, tuple)):
                fill = np.array(self.fill_value, dtype=img.dtype)
            else:
                fill = np.full(3, self.fill_value, dtype=img.dtype)

            img_masked = (img * mask_3d + fill * (1 - mask_3d)).astype(img.dtype)
        else:
            # Grayscale image
            img_masked = (img * mask + self.fill_value * (1 - mask)).astype(img.dtype)

        results['img'] = img_masked
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'radius_ratio={self.radius_ratio}, '
                f'soft_edge={self.soft_edge}, '
                f'edge_width={self.edge_width}, '
                f'fill_value={self.fill_value})')


@TRANSFORMS.register_module()
class RandomVignette(BaseTransform):
    """Apply random vignetting augmentation to training images.

    This can be used to augment training images to match the vignetting
    characteristics of test images, reducing domain gap.

    Args:
        prob (float): Probability of applying vignetting. Default: 0.5
        strength_range (tuple): Range of vignetting strength (min, max).
            Default: (0.3, 0.7)
        radius_range (tuple): Range of vignetting radius ratio (min, max).
            Default: (0.7, 1.0)
    """

    def __init__(self,
                 prob: float = 0.5,
                 strength_range: tuple = (0.3, 0.7),
                 radius_range: tuple = (0.7, 1.0)):
        self.prob = prob
        self.strength_range = strength_range
        self.radius_range = radius_range

    def transform(self, results: dict) -> dict:
        """Apply random vignetting to image.

        Args:
            results: Dict containing 'img' key with image array.

        Returns:
            Modified results dict with vignetting applied.
        """
        if np.random.random() > self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]

        # Random parameters
        strength = np.random.uniform(*self.strength_range)
        radius = np.random.uniform(*self.radius_range)

        # Create vignette mask
        center_y, center_x = h / 2, w / 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist_normalized = dist / max_dist

        # Vignette falloff (darker at edges)
        vignette = 1 - strength * np.clip((dist_normalized - radius) / (1 - radius), 0, 1) ** 2

        # Apply vignetting
        if len(img.shape) == 3:
            vignette = vignette[:, :, np.newaxis]

        img_vignetted = (img * vignette).astype(img.dtype)
        results['img'] = img_vignetted

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'prob={self.prob}, '
                f'strength_range={self.strength_range}, '
                f'radius_range={self.radius_range})')
