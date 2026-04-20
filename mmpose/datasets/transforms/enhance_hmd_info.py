"""Transform to enhance HMD info with additional dimensions from keypoint3d.

This module provides transforms that add extra information to the 9-dim HMD info
by extracting coordinates from keypoint3d.

IMPORTANT: Ground reference is computed using BODY-AXIS method (Vector A),
NOT coordinate axes. In camera coordinates, no axis corresponds to "up/down".
We use the body's skeletal structure to define the downward direction.

Body-Axis Method:
    1. Vector A = direction from root (Spine2) toward pelvis center
    2. Ground = farthest toe projection along Vector A
    3. Heights = projection distances along Vector A

Working Options:
    - hands_y: Add hands Y relative to head (11 dims) [LEGACY - uses Y axis]
    - torso_reference: Add torso Y + head-torso distance (11 dims) [LEGACY]
    - ground_reference: Add root height from ground + head-torso distance (11 dims)
      ※ Uses body-axis method for correct ground estimation
    - head_from_ground: Add only root height from ground (10 dims)
      ※ Uses body-axis method
    - hand_from_ground: Add hand heights from ground (11 dims)
      ※ Uses body-axis method
    - both_from_ground: Add root + hand heights from ground (12 dims)
      ※ Most complete ground-based info, uses body-axis method
    - relative_heights: Add heights relative to torso (12 dims) [LEGACY]
"""

import numpy as np
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnhanceHMDInfo:
    """Enhance HMD info with additional dimensions from keypoint3d.

    This transform adds extra information to the existing 9-dim HMD info
    by computing additional values from the 3D keypoints.

    IMPORTANT: For ground-related modes, we use the BODY-AXIS method:
    - In camera coordinates, NO axis corresponds to "up" or "down"
    - We define "downward" using the body's skeletal structure
    - Vector A: from root (Spine2) toward pelvis center
    - Ground: farthest toe projection along Vector A
    - Heights: perpendicular distance to ground plane along Vector A

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
            - 'ground_reference': Add root height + head-to-torso distance (11 dims)
              Uses body-axis method for accurate ground estimation.
            - 'head_from_ground': Add only root height from ground (10 dims)
              Simplest option using body-axis method.
            - 'hand_from_ground': Add left/right hand heights from ground (11 dims)
              Hand heights measured along body axis.
            - 'both_from_ground': Add root + left/right hand heights (12 dims)
              Most complete ground-based info using body-axis method.
            - 'relative_heights': Add heights relative to torso center (12 dims)

    Required Keys:
        - hmd_info: (1, 9) existing HMD info
        - keypoint3d: (1, 16, 3) 3D keypoints

    Modified Keys:
        - hmd_info: enhanced HMD info with additional dimensions
    """

    # Keypoint indices for xR-EgoPose (16 joints)
    # 0:Spine2(root), 1:Head, 2:LeftArm, 3:LeftForeArm, 4:LeftHand,
    # 5:RightArm, 6:RightForeArm, 7:RightHand,
    # 8:LeftUpLeg, 9:LeftLeg, 10:LeftFoot, 11:LeftToeBase,
    # 12:RightUpLeg, 13:RightLeg, 14:RightFoot, 15:RightToeBase
    ROOT_IDX = 0          # Spine2 (root)
    HEAD_IDX = 0          # For HMD info, we use root as "head" reference
    LEFT_HAND_IDX = 4
    RIGHT_HAND_IDX = 7
    LEFT_UPLEG_IDX = 8    # LeftUpLeg (for pelvis center)
    RIGHT_UPLEG_IDX = 12  # RightUpLeg (for pelvis center)
    LEFT_FOOT_IDX = 10
    RIGHT_FOOT_IDX = 14
    LEFT_TOE_IDX = 11     # For ground reference
    RIGHT_TOE_IDX = 15    # For ground reference

    def __init__(self, mode: str = 'torso_reference'):
        valid_modes = [
            'hands_y', 'torso_reference', 'ground_reference',
            'head_from_ground', 'hand_from_ground', 'both_from_ground',
            'relative_heights'
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        self.mode = mode

    def _compute_body_axis_ground(self, p3d: np.ndarray) -> dict:
        """Compute ground reference using body-axis method (Vector A).

        In camera coordinates, no axis corresponds to "up/down".
        We use the body's skeletal structure to define the downward direction:
        - Vector A: from root (Spine2) toward pelvis center
        - Ground: farthest toe projection along Vector A

        Args:
            p3d: (16, 3) array of 3D keypoints

        Returns:
            dict with:
                - vec_a_unit: unit vector of body axis (downward direction)
                - ground_ref_proj: ground projection distance along Vector A
                - root_from_ground: root height from ground (along body axis)
        """
        root = p3d[self.ROOT_IDX]
        left_upleg = p3d[self.LEFT_UPLEG_IDX]
        right_upleg = p3d[self.RIGHT_UPLEG_IDX]
        left_toe = p3d[self.LEFT_TOE_IDX]
        right_toe = p3d[self.RIGHT_TOE_IDX]

        # Step 1: Compute pelvis center (midpoint of hip joints)
        pelvis_center = (left_upleg + right_upleg) / 2

        # Step 2: Vector A = direction from root toward pelvis (body "downward")
        vec_a = pelvis_center - root
        vec_a_norm = np.linalg.norm(vec_a)
        if vec_a_norm > 1e-6:
            vec_a_unit = vec_a / vec_a_norm
        else:
            # Fallback if pelvis is at same position as root
            vec_a_unit = np.array([0, 0, 1], dtype=np.float32)

        # Step 3: Project toes onto Vector A to find ground reference
        # Ground = farthest toe along body axis
        left_toe_proj = np.dot(left_toe - root, vec_a_unit)
        right_toe_proj = np.dot(right_toe - root, vec_a_unit)
        ground_ref_proj = max(left_toe_proj, right_toe_proj)

        # Step 4: Root height from ground (along body axis)
        # Root projection is 0 (it's the origin of our vector)
        root_from_ground = ground_ref_proj  # = ground_ref_proj - 0

        return {
            'vec_a_unit': vec_a_unit,
            'ground_ref_proj': ground_ref_proj,
            'root_from_ground': root_from_ground,
            'pelvis_center': pelvis_center,
        }

    def __call__(self, results: dict) -> dict:
        """Enhance HMD info with additional dimensions.

        Args:
            results: Dict containing 'hmd_info' and 'keypoint3d'

        Returns:
            Dict with enhanced 'hmd_info'
        """
        hmd_info = results['hmd_info']  # (1, 9)
        keypoint3d = results['keypoint3d']  # (1, 16, 3)

        # Extract keypoints
        p3d = keypoint3d[0]  # (16, 3)
        root = p3d[self.ROOT_IDX]
        right_hand = p3d[self.RIGHT_HAND_IDX]
        left_hand = p3d[self.LEFT_HAND_IDX]
        left_upleg = p3d[self.LEFT_UPLEG_IDX]
        right_upleg = p3d[self.RIGHT_UPLEG_IDX]

        # Compute pelvis/torso reference
        pelvis_center = (left_upleg + right_upleg) / 2

        # Compute additional features based on mode
        if self.mode == 'hands_y':
            # LEGACY: Hands Y relative to root (uses Y axis directly)
            # Kept for backward compatibility
            extra = np.array([right_hand[1], left_hand[1]], dtype=np.float32)

        elif self.mode == 'torso_reference':
            # LEGACY: Torso Y + head-to-torso distance
            head_torso_dist = np.linalg.norm(root - pelvis_center)
            extra = np.array([pelvis_center[1], head_torso_dist], dtype=np.float32)

        elif self.mode == 'ground_reference':
            # CORRECT: Body-axis method for ground reference
            ground_info = self._compute_body_axis_ground(p3d)
            root_from_ground = ground_info['root_from_ground']
            head_torso_dist = np.linalg.norm(root - pelvis_center)
            extra = np.array([root_from_ground, head_torso_dist], dtype=np.float32)

        elif self.mode == 'head_from_ground':
            # CORRECT: Body-axis method - root height from ground
            ground_info = self._compute_body_axis_ground(p3d)
            root_from_ground = ground_info['root_from_ground']
            extra = np.array([root_from_ground], dtype=np.float32)

        elif self.mode == 'hand_from_ground':
            # CORRECT: Body-axis method - hand heights from ground
            ground_info = self._compute_body_axis_ground(p3d)
            vec_a_unit = ground_info['vec_a_unit']
            ground_ref_proj = ground_info['ground_ref_proj']

            # Project hands onto body axis
            left_hand_proj = np.dot(left_hand - root, vec_a_unit)
            right_hand_proj = np.dot(right_hand - root, vec_a_unit)

            # Hand heights from ground (along body axis)
            left_hand_from_ground = ground_ref_proj - left_hand_proj
            right_hand_from_ground = ground_ref_proj - right_hand_proj

            extra = np.array([left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'both_from_ground':
            # CORRECT: Body-axis method - root + both hands from ground
            ground_info = self._compute_body_axis_ground(p3d)
            vec_a_unit = ground_info['vec_a_unit']
            ground_ref_proj = ground_info['ground_ref_proj']
            root_from_ground = ground_info['root_from_ground']

            # Project hands onto body axis
            left_hand_proj = np.dot(left_hand - root, vec_a_unit)
            right_hand_proj = np.dot(right_hand - root, vec_a_unit)

            # Hand heights from ground (along body axis)
            left_hand_from_ground = ground_ref_proj - left_hand_proj
            right_hand_from_ground = ground_ref_proj - right_hand_proj

            extra = np.array([root_from_ground, left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'relative_heights':
            # LEGACY: Heights relative to torso center (uses Y axis)
            head_rel_y = root[1] - pelvis_center[1]
            right_rel_y = right_hand[1] - pelvis_center[1]
            left_rel_y = left_hand[1] - pelvis_center[1]
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
        'ground_reference': 11,  # 9 + 2 (root_from_ground + head_torso_dist)
        'head_from_ground': 10,  # 9 + 1 (root_from_ground only)
        'hand_from_ground': 11,  # 9 + 2 (left_hand + right_hand from ground)
        'both_from_ground': 12,  # 9 + 3 (root + left_hand + right_hand from ground)
        'relative_heights': 12,  # 9 + 3
    }
    return sizes.get(mode, 9)


@TRANSFORMS.register_module()
class ZeroHMDInfo:
    """Replace HMD info with zeros (vision-only ablation).

    Use this transform for fair comparison with methods that do not
    use HMD/controller tracking input. The HMD info tensor is kept
    at the same size but filled with zeros, so the model architecture
    remains unchanged.

    Args:
        hmd_dim (int): Expected HMD info dimension. Default: 9.
    """

    def __init__(self, hmd_dim: int = 9):
        self.hmd_dim = hmd_dim

    def __call__(self, results: dict) -> dict:
        results['hmd_info'] = np.zeros((1, self.hmd_dim), dtype=np.float32)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hmd_dim={self.hmd_dim})'
