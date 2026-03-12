"""Transform to enhance HMD info with additional dimensions from keypoint3d for Mo2Cap2 dataset.

Mo2Cap2 has 15 joints with different indices than EgoPose (16 joints):
- Mo2Cap2 Joint indices:
  - 0: Neck (root)
  - 1-3: RightArm, RightForeArm, RightHand
  - 4-6: LeftArm, LeftForeArm, LeftHand
  - 7-10: RightUpLeg, RightLeg, RightFoot, RightToeBase
  - 11-14: LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase

IMPORTANT: Since 3D joints are in CAMERA coordinates (not world coordinates),
none of the X/Y/Z axes directly correspond to the user's height.
We use a BODY-RELATIVE direction for ground reference:

    Vector a = direction from Neck[0] toward pelvis center
    pelvis_center = average(L.UpLeg[11], R.UpLeg[7])

Ground reference is the farthest toe (L.ToeBase[14] or R.ToeBase[10])
projected onto vector a.

Working Options:
    - head_from_ground: Add only neck height from ground (10 dims)
    - hand_from_ground: Add hand heights from ground (11 dims)
    - both_from_ground: Add neck + hand heights from ground (12 dims)
"""

import numpy as np
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnhanceHMDInfo_Mo2Cap2:
    """Enhance HMD info with body-relative ground reference for Mo2Cap2.

    This transform adds ground reference information using the body's own
    orientation, which works regardless of camera orientation.

    Method:
        1. Vector a = normalize(pelvis_center - neck)
           where pelvis_center = average(L.UpLeg, R.UpLeg)
        2. Ground reference = farthest toe projection onto vector a
           Compare L.ToeBase[14] vs R.ToeBase[10]
        3. Heights are distances along vector a to the ground plane

    The base 9-dim HMD info contains:
        - right_local (3): right hand in local coordinate system
        - left_local (3): left hand in local coordinate system
        - hand_distance (1): distance between hands
        - right_distance (1): head to right hand distance
        - left_distance (1): head to left hand distance

    Args:
        mode (str): Enhancement mode. Options:
            - 'head_from_ground': Add only neck height from ground (10 dims)
            - 'hand_from_ground': Add left/right hand heights from ground (11 dims)
            - 'both_from_ground': Add neck + left/right hand heights (12 dims)

    Required Keys:
        - hmd_info: (1, 9) existing HMD info
        - keypoint3d: (1, 15, 3) 3D keypoints (root-relative, Neck at origin)

    Modified Keys:
        - hmd_info: enhanced HMD info with additional dimensions
    """

    # Mo2Cap2 Joint indices
    NECK_IDX = 0          # Root joint (at origin after root-relative transform)
    RIGHT_HAND_IDX = 3    # RightHand
    LEFT_HAND_IDX = 6     # LeftHand
    RIGHT_UPLEG_IDX = 7   # RightUpLeg (for pelvis center)
    LEFT_UPLEG_IDX = 11   # LeftUpLeg (for pelvis center)
    RIGHT_TOE_IDX = 10    # RightToeBase (for ground reference)
    LEFT_TOE_IDX = 14     # LeftToeBase (for ground reference)

    def __init__(self, mode: str = 'both_from_ground'):
        valid_modes = [
            'head_from_ground', 'hand_from_ground', 'both_from_ground'
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        self.mode = mode

    def _compute_body_relative_ground(self, p3d: np.ndarray) -> dict:
        """Compute ground reference using body-relative direction.

        Args:
            p3d: (15, 3) 3D keypoints (root-relative, Neck at origin)

        Returns:
            dict with vec_a_unit, ground_ref_proj, and joint projections
        """
        neck = p3d[self.NECK_IDX]  # [0, 0, 0] after root-relative
        left_upleg = p3d[self.LEFT_UPLEG_IDX]
        right_upleg = p3d[self.RIGHT_UPLEG_IDX]
        left_hand = p3d[self.LEFT_HAND_IDX]
        right_hand = p3d[self.RIGHT_HAND_IDX]
        left_toe = p3d[self.LEFT_TOE_IDX]
        right_toe = p3d[self.RIGHT_TOE_IDX]

        # Compute pelvis center (average of UpLegs)
        pelvis_center = (left_upleg + right_upleg) / 2

        # Vector a: direction from Neck to pelvis (body "downward" direction)
        vec_a = pelvis_center - neck  # neck is [0,0,0], so this equals pelvis_center
        vec_a_norm = np.linalg.norm(vec_a)

        if vec_a_norm < 1e-8:
            # Fallback: use a default direction if pelvis is at neck
            vec_a_unit = np.array([0, 0, 1], dtype=np.float32)
        else:
            vec_a_unit = vec_a / vec_a_norm

        # Project toes onto vector a to find ground reference
        # Projection: proj = dot(point - neck, vec_a_unit) = dot(point, vec_a_unit)
        left_toe_proj = np.dot(left_toe, vec_a_unit)
        right_toe_proj = np.dot(right_toe, vec_a_unit)

        # Ground reference is the farthest toe along vector a
        ground_ref_proj = max(left_toe_proj, right_toe_proj)

        # Project hands onto vector a
        left_hand_proj = np.dot(left_hand, vec_a_unit)
        right_hand_proj = np.dot(right_hand, vec_a_unit)

        return {
            'vec_a_unit': vec_a_unit,
            'ground_ref_proj': ground_ref_proj,
            'left_hand_proj': left_hand_proj,
            'right_hand_proj': right_hand_proj,
        }

    def __call__(self, results: dict) -> dict:
        """Enhance HMD info with body-relative ground reference.

        Args:
            results: Dict containing 'hmd_info' and 'keypoint3d'

        Returns:
            Dict with enhanced 'hmd_info'
        """
        hmd_info = results['hmd_info']  # (1, 9)
        keypoint3d = results['keypoint3d']  # (1, 15, 3)

        # Extract keypoints (root-relative: Neck at origin)
        p3d = keypoint3d[0]  # (15, 3)

        # Compute body-relative ground reference
        ground_data = self._compute_body_relative_ground(p3d)
        ground_ref_proj = ground_data['ground_ref_proj']

        # Compute ground info based on mode
        if self.mode == 'head_from_ground':
            # Neck height from ground (along body axis)
            # Since neck is at origin, neck_proj = 0, so height = ground_ref_proj
            neck_from_ground = ground_ref_proj
            extra = np.array([neck_from_ground], dtype=np.float32)

        elif self.mode == 'hand_from_ground':
            # Hand heights from ground (along body axis)
            left_hand_from_ground = ground_ref_proj - ground_data['left_hand_proj']
            right_hand_from_ground = ground_ref_proj - ground_data['right_hand_proj']
            extra = np.array([left_hand_from_ground, right_hand_from_ground], dtype=np.float32)

        elif self.mode == 'both_from_ground':
            # Complete ground-based info: neck + both hands from ground
            neck_from_ground = ground_ref_proj
            left_hand_from_ground = ground_ref_proj - ground_data['left_hand_proj']
            right_hand_from_ground = ground_ref_proj - ground_data['right_hand_proj']
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
