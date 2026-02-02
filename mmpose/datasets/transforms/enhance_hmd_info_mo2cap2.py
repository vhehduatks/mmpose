"""Transform to enhance HMD info with additional dimensions from keypoint3d for Mo2Cap2 dataset.

Mo2Cap2 has 15 joints with different indices than EgoPose (16 joints):
- Mo2Cap2 Joint indices:
  - 0: Neck (root)
  - 1-3: RightArm, RightForeArm, RightHand
  - 4-6: LeftArm, LeftForeArm, LeftHand
  - 7-10: RightUpLeg, RightLeg, RightFoot, RightToeBase
  - 11-14: LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase

IMPORTANT: keypoint3d is ROOT-RELATIVE (Neck at origin), so neck_y = 0 always.
Ground reference is estimated from feet positions (lowest Y value).

Working Options:
    - head_from_ground: Add only neck height from ground (10 dims)
    - hand_from_ground: Add hand heights from ground (11 dims)
    - both_from_ground: Add neck + hand heights from ground (12 dims)
"""

import numpy as np
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnhanceHMDInfo_Mo2Cap2:
    """Enhance HMD info with additional dimensions from keypoint3d for Mo2Cap2.

    This transform adds extra information to the existing 9-dim HMD info
    by computing additional values from the 3D keypoints.

    NOTE: keypoint3d is root-relative (Neck at origin), so neck_y = 0 always.
    Only features that are meaningful in the root-relative coordinate system
    are supported.

    The base 9-dim HMD info contains:
        - right_local (3): right hand in local coordinate system
        - left_local (3): left hand in local coordinate system
        - hand_distance (1): distance between hands
        - right_distance (1): head to right hand distance
        - left_distance (1): head to left hand distance

    Args:
        mode (str): Enhancement mode. Options:
            - 'head_from_ground': Add only neck height from ground (10 dims)
              This is what HMD tracking systems can directly measure.
            - 'hand_from_ground': Add left/right hand heights from ground (11 dims)
              Hand positions relative to ground plane.
            - 'both_from_ground': Add neck + left/right hand heights (12 dims)
              Most complete ground-based info for HMD deployment.

    Required Keys:
        - hmd_info: (1, 9) existing HMD info
        - keypoint3d: (1, 15, 3) 3D keypoints (root-relative)

    Modified Keys:
        - hmd_info: enhanced HMD info with additional dimensions
    """

    # Mo2Cap2 Joint indices
    NECK_IDX = 0          # Root joint (always at origin)
    RIGHT_HAND_IDX = 3    # RightHand
    LEFT_HAND_IDX = 6     # LeftHand
    RIGHT_FOOT_IDX = 9    # RightFoot
    LEFT_FOOT_IDX = 13    # LeftFoot
    RIGHT_TOE_IDX = 10    # RightToeBase
    LEFT_TOE_IDX = 14     # LeftToeBase

    def __init__(self, mode: str = 'both_from_ground'):
        valid_modes = [
            'head_from_ground', 'hand_from_ground', 'both_from_ground'
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        self.mode = mode

    def __call__(self, results: dict) -> dict:
        """Enhance HMD info with additional dimensions.

        Args:
            results: Dict containing 'hmd_info' and 'keypoint3d'

        Returns:
            Dict with enhanced 'hmd_info'
        """
        hmd_info = results['hmd_info']  # (1, 9)
        keypoint3d = results['keypoint3d']  # (1, 15, 3)

        # Extract keypoints (root-relative: Neck at origin)
        p3d = keypoint3d[0]  # (15, 3)
        neck = p3d[self.NECK_IDX]  # Always [0, 0, 0]
        right_hand = p3d[self.RIGHT_HAND_IDX]
        left_hand = p3d[self.LEFT_HAND_IDX]
        right_foot = p3d[self.RIGHT_FOOT_IDX]
        left_foot = p3d[self.LEFT_FOOT_IDX]

        # Estimate ground from feet positions (lowest Y value)
        # In root-relative coords, feet Y are negative (below neck)
        ground_y = min(left_foot[1], right_foot[1])

        # Compute additional features based on mode
        if self.mode == 'head_from_ground':
            # Simplest ground-based feature: only neck height from ground
            # This is what HMD tracking systems can directly measure
            # neck_from_ground = 0 - ground_y = -ground_y (positive value)
            neck_from_ground = -ground_y
            extra = np.array([neck_from_ground], dtype=np.float32)

        elif self.mode == 'hand_from_ground':
            # Hand heights from ground (controller tracking)
            # Useful for understanding arm/hand positions relative to floor
            left_hand_from_ground = left_hand[1] - ground_y
            right_hand_from_ground = right_hand[1] - ground_y
            extra = np.array([left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'both_from_ground':
            # Complete ground-based info: neck + both hands from ground
            # Most comprehensive for HMD deployment scenarios
            neck_from_ground = -ground_y
            left_hand_from_ground = left_hand[1] - ground_y
            right_hand_from_ground = right_hand[1] - ground_y
            extra = np.array([neck_from_ground, left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        # Concatenate to hmd_info
        # hmd_info shape: (1, 9) -> (1, 9+len(extra))
        enhanced_hmd = np.concatenate([hmd_info[0], extra])[np.newaxis, :]
        results['hmd_info'] = enhanced_hmd.astype(np.float32)

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mode={self.mode!r})'


# Convenience function to get output size for each mode
def get_enhanced_hmd_size_mo2cap2(mode: str) -> int:
    """Get the output HMD info size for a given enhancement mode.

    Args:
        mode: Enhancement mode name

    Returns:
        int: Total HMD info dimensions after enhancement
    """
    sizes = {
        'head_from_ground': 10,  # 9 + 1 (neck_from_ground only)
        'hand_from_ground': 11,  # 9 + 2 (left_hand + right_hand from ground)
        'both_from_ground': 12,  # 9 + 3 (neck + left_hand + right_hand from ground)
    }
    return sizes.get(mode, 9)
