"""Transform to enhance HMD info with additional dimensions from keypoint3d.

This module provides transforms that add extra information to the 9-dim HMD info
by extracting coordinates from keypoint3d.

IMPORTANT: keypoint3d is ROOT-RELATIVE (head at origin), so head_y = 0 always.
Only options that work with root-relative data are supported.

Working Options:
    - hands_y: Add hands Y relative to head (11 dims)
    - torso_reference: Add torso Y + head-torso distance (11 dims)
    - ground_reference: Add head height from ground + head-torso distance (11 dims)
      ※ Ground is estimated from feet positions (realistic for HMD deployment)
    - head_from_ground: Add only head height from ground (10 dims)
      ※ Simplest ground-based feature - just head height
    - hand_from_ground: Add hand heights from ground (11 dims)
      ※ Left and right hand heights from estimated ground
    - both_from_ground: Add head + hand heights from ground (12 dims)
      ※ Head height + left hand height + right hand height from ground
    - relative_heights: Add heights relative to torso (12 dims)
"""

import numpy as np
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnhanceHMDInfo:
    """Enhance HMD info with additional dimensions from keypoint3d.

    This transform adds extra information to the existing 9-dim HMD info
    by computing additional values from the 3D keypoints.

    NOTE: keypoint3d is root-relative (head at origin), so head_y = 0 always.
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
            - 'hands_y': Add right hand Y, left hand Y relative to head (11 dims)
            - 'torso_reference': Add torso Y + head-to-torso distance (11 dims)
            - 'ground_reference': Add head height + head-to-torso distance (11 dims)
              Ground is estimated from min(feet_y). This is closer to real HMD data.
            - 'head_from_ground': Add only head height from ground (10 dims)
              Simplest option - just one additional feature.
            - 'hand_from_ground': Add left/right hand heights from ground (11 dims)
              Hand positions relative to ground plane.
            - 'both_from_ground': Add head + left/right hand heights (12 dims)
              Most complete ground-based info for HMD deployment.
            - 'relative_heights': Add heights relative to torso center (12 dims)

    Required Keys:
        - hmd_info: (1, 9) existing HMD info
        - keypoint3d: (1, 16, 3) 3D keypoints (root-relative)

    Modified Keys:
        - hmd_info: enhanced HMD info with additional dimensions
    """

    # Keypoint indices (from config.skel order)
    # 0:Head, 1:Neck, 2:LeftArm, 3:LeftForeArm, 4:LeftHand,
    # 5:RightArm, 6:RightForeArm, 7:RightHand,
    # 8:LeftUpLeg, 9:LeftLeg, 10:LeftFoot, 11:LeftToeBase,
    # 12:RightUpLeg, 13:RightLeg, 14:RightFoot, 15:RightToeBase
    HEAD_IDX = 0
    LEFT_HAND_IDX = 4
    RIGHT_HAND_IDX = 7
    LEFT_HIP_IDX = 8      # LeftUpLeg (pelvis)
    RIGHT_HIP_IDX = 12    # RightUpLeg (pelvis)
    LEFT_FOOT_IDX = 10
    RIGHT_FOOT_IDX = 14
    LEFT_TOE_IDX = 11
    RIGHT_TOE_IDX = 15

    def __init__(self, mode: str = 'torso_reference'):
        valid_modes = [
            'hands_y', 'torso_reference', 'ground_reference',
            'head_from_ground', 'hand_from_ground', 'both_from_ground',
            'relative_heights'
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
        keypoint3d = results['keypoint3d']  # (1, 16, 3)

        # Extract keypoints (root-relative: head at origin)
        p3d = keypoint3d[0]  # (16, 3)
        head = p3d[self.HEAD_IDX]  # Always [0, 0, 0]
        right_hand = p3d[self.RIGHT_HAND_IDX]
        left_hand = p3d[self.LEFT_HAND_IDX]
        right_hip = p3d[self.RIGHT_HIP_IDX]
        left_hip = p3d[self.LEFT_HIP_IDX]

        # Compute torso reference (average of hips/pelvis)
        torso_ref = (right_hip + left_hip) / 2

        # Compute additional features based on mode
        if self.mode == 'hands_y':
            # Hands Y relative to head (head_y = 0, so this is just hands Y)
            # Tells vertical position of hands relative to head
            extra = np.array([right_hand[1], left_hand[1]], dtype=np.float32)

        elif self.mode == 'torso_reference':
            # Torso Y (relative to head) + head-to-torso distance
            # torso_y is negative (torso below head)
            head_torso_dist = np.linalg.norm(head - torso_ref)
            extra = np.array([torso_ref[1], head_torso_dist], dtype=np.float32)

        elif self.mode == 'ground_reference':
            # Estimate ground from feet positions (more realistic for HMD deployment)
            # In root-relative coords, feet Y are negative (below head)
            left_foot = p3d[self.LEFT_FOOT_IDX]
            right_foot = p3d[self.RIGHT_FOOT_IDX]

            # Ground is estimated as the lowest foot Y (most negative)
            ground_y = min(left_foot[1], right_foot[1])

            # Heights from ground (what real HMD tracking provides)
            # head_from_ground = 0 - ground_y = -ground_y (positive value)
            head_from_ground = -ground_y
            # For second feature, use body scale (head-torso distance) like torso_ref
            head_torso_dist = np.linalg.norm(head - torso_ref)
            extra = np.array([head_from_ground, head_torso_dist], dtype=np.float32)

        elif self.mode == 'head_from_ground':
            # Simplest ground-based feature: only head height from ground
            # This is what HMD tracking systems can directly measure
            left_foot = p3d[self.LEFT_FOOT_IDX]
            right_foot = p3d[self.RIGHT_FOOT_IDX]
            ground_y = min(left_foot[1], right_foot[1])
            head_from_ground = -ground_y
            extra = np.array([head_from_ground], dtype=np.float32)

        elif self.mode == 'hand_from_ground':
            # Hand heights from ground (controller tracking)
            # Useful for understanding arm/hand positions relative to floor
            left_foot = p3d[self.LEFT_FOOT_IDX]
            right_foot = p3d[self.RIGHT_FOOT_IDX]
            ground_y = min(left_foot[1], right_foot[1])
            left_hand_from_ground = left_hand[1] - ground_y
            right_hand_from_ground = right_hand[1] - ground_y
            extra = np.array([left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'both_from_ground':
            # Complete ground-based info: head + both hands from ground
            # Most comprehensive for HMD deployment scenarios
            left_foot = p3d[self.LEFT_FOOT_IDX]
            right_foot = p3d[self.RIGHT_FOOT_IDX]
            ground_y = min(left_foot[1], right_foot[1])
            head_from_ground = -ground_y
            left_hand_from_ground = left_hand[1] - ground_y
            right_hand_from_ground = right_hand[1] - ground_y
            extra = np.array([head_from_ground, left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'relative_heights':
            # Heights relative to torso center
            # head_rel_y = 0 - torso_y = -torso_y (positive, head above torso)
            # hands relative to torso
            head_rel_y = head[1] - torso_ref[1]
            right_rel_y = right_hand[1] - torso_ref[1]
            left_rel_y = left_hand[1] - torso_ref[1]
            extra = np.array([head_rel_y, right_rel_y, left_rel_y], dtype=np.float32)

        # Concatenate to hmd_info
        # hmd_info shape: (1, 9) -> (1, 9+len(extra))
        enhanced_hmd = np.concatenate([hmd_info[0], extra])[np.newaxis, :]
        results['hmd_info'] = enhanced_hmd.astype(np.float32)

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mode={self.mode!r})'


# Convenience function to get output size for each mode
def get_enhanced_hmd_size(mode: str) -> int:
    """Get the output HMD info size for a given enhancement mode.

    Args:
        mode: Enhancement mode name

    Returns:
        int: Total HMD info dimensions after enhancement
    """
    sizes = {
        'hands_y': 11,           # 9 + 2
        'torso_reference': 11,   # 9 + 2
        'ground_reference': 11,  # 9 + 2 (head_from_ground + head_torso_dist)
        'head_from_ground': 10,  # 9 + 1 (head_from_ground only)
        'hand_from_ground': 11,  # 9 + 2 (left_hand + right_hand from ground)
        'both_from_ground': 12,  # 9 + 3 (head + left_hand + right_hand from ground)
        'relative_heights': 12,  # 9 + 3
    }
    return sizes.get(mode, 9)
