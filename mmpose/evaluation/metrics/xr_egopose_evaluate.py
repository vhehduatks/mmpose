# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""
xR-EgoPose Evaluation Module

Evaluation protocol for xR-EgoPose egocentric pose estimation dataset.
- MPJPE (Mean Per Joint Position Error) in mm
- Per-action breakdown
- Upper/Lower body metrics
"""
import re
from abc import ABC, abstractmethod

import numpy as np


# ============================================================================
# Constants
# ============================================================================

# xR-EgoPose joint order (16 joints)
XREGOPOSE_JOINTS = [
    'head',           # 0
    'neck',           # 1
    'right_shoulder', # 2
    'right_elbow',    # 3
    'right_wrist',    # 4
    'left_shoulder',  # 5
    'left_elbow',     # 6
    'left_wrist',     # 7
    'right_hip',      # 8
    'right_knee',     # 9
    'right_ankle',    # 10
    'left_hip',       # 11
    'left_knee',      # 12
    'left_ankle',     # 13
    'pelvis',         # 14
    'spine',          # 15
]

# Upper body: head, neck, shoulders, elbows, wrists, spine
XREGOPOSE_UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 15]

# Lower body: hips, knees, ankles, pelvis
XREGOPOSE_LOWER_BODY_INDICES = [8, 9, 10, 11, 12, 13, 14]

# Action names in xR-EgoPose dataset (after removing _mixamo_com suffix)
ACTION_NAMES = [
    'Angry_Point',
    'Arm_Gesture',
    'Beckoning',
    'Charge',
    'Clapping',
    'Counting__1_',
    'Crazy_Gesture',
    'Dismissing_Gesture',
    'Fist_Pump',
    'Golf_Putt_Failure',
    'Golf_Putt_Victory__1_',
    'Hand_Raising',
    'Hands_Forward_Gesture',
    'Happy_Hand_Gesture',
    'Insult',
    'Loser',
    'No',
    'Pain_Gesture',
    'Petting',
    'Petting_Animal',
    'Pointing',
    'Pointing_Gesture',
    'Quick_Formal_Bow',
    'Rallying',
    'Reaching_Out',
    'Revealing_Dice',
    'Shaking_Hands_2',
    'Sitting_Disapproval',
    'Standing_1H_Magic_Attack_01',
    'Standing_Greeting',
    'Strong_Gesture',
    'Surprised',
    'Talking',
    'Taunt_Gesture',
    'Terrified',
    'Thinking',
    'Tpose_Take_001',
    'Weight_Shift_Gesture',
    'anim_Clip1',
    'lower_stretching',
    'upper_stretching',
    'walking',
]

# Set for fast lookup
ACTION_SET = set(ACTION_NAMES)

__all__ = [
    "ACTION_NAMES", "ACTION_SET",
    "XREGOPOSE_JOINTS", "XREGOPOSE_UPPER_BODY_INDICES", "XREGOPOSE_LOWER_BODY_INDICES",
    "EvalBody", "EvalUpperBody", "EvalLowerBody",
    "map_action_name", "compute_mpjpe",
]


# ============================================================================
# Utility functions
# ============================================================================

def map_action_name(name) -> str:
    """Map raw action name to standardized action category.

    Removes _mixamo_com suffix and variants, then checks if it's a known action.

    Args:
        name: Raw action name from dataset (e.g., 'Clapping_mixamo_com')
              Can be str or bytes.

    Returns:
        Mapped action name or 'All' if not recognized
    """
    # Ensure name is a string
    if isinstance(name, (bytes, np.bytes_)):
        name = name.decode('utf-8')
    elif not isinstance(name, str):
        # Handle numpy scalars and other types
        name = str(name)
        if name.startswith("b'") or name.startswith('b"'):
            # String representation of bytes, decode it
            name = name[2:-1]

    # Remove _mixamo_com and variants (e.g., _mixamo_com1)
    suffix = re.findall(r'_mixamo_com.*', name)
    if suffix:
        name = name.replace(suffix[0], '')

    # Check xRegopose action set first
    if name in ACTION_SET:
        return name

    # Kinect action grouping: map specific actions to categories
    # e.g., 'Dancing1', 'Dancing2' -> 'Dancing'
    # e.g., 'Gaming-Archery', 'Gaming-Boxing' -> 'Gaming'
    # e.g., 'Workout-BicelCurl', 'Workout-FrontRaise' -> 'Workout'
    kinect_prefix_map = {
        'Dancing': 'Dancing',
        'Gaming': 'Gaming',
        'Greeting': 'Greeting',
        'Reacting': 'Reacting',
        'Workout': 'Workout',
    }
    for prefix, category in kinect_prefix_map.items():
        if name.startswith(prefix):
            return category

    # Return raw name for other Kinect actions (e.g., Patting, Talking, Walking, UpperStreching)
    return name if name != '' else 'All'


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray,
                  joint_indices: list = None) -> float:
    """Compute Mean Per Joint Position Error (MPJPE).

    Args:
        pred: Predicted 3D keypoints, shape (N, 16, 3) or (16, 3)
        gt: Ground truth 3D keypoints, same shape as pred
        joint_indices: Optional list of joint indices to compute error for.
                      If None, uses all joints.

    Returns:
        MPJPE in the same unit as input (typically mm)
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    if joint_indices is not None:
        pred = pred[:, joint_indices, :]
        gt = gt[:, joint_indices, :]

    # Compute per-joint Euclidean distance
    errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))  # (N, J)

    # Mean over joints and samples
    return float(np.mean(errors))


# ============================================================================
# Evaluation classes
# ============================================================================

class BaseEval(ABC):
    """Base evaluation class for xR-EgoPose."""

    def __init__(self):
        super().__init__()
        self.error = {'All': []}
        # Initialize all action categories
        for action in ACTION_NAMES:
            self.error[action] = []

    def _add_error(self, err: float, action: str = None):
        """Add error to tracking.

        Args:
            err: MPJPE error value
            action: Action name (will be mapped to category)
        """
        self.error['All'].append(err)
        if action:
            mapped_action = map_action_name(action)
            if mapped_action in self.error:
                self.error[mapped_action].append(err)

    def get_results(self) -> dict:
        """Get evaluation results.

        Returns:
            Dictionary with per-action results containing:
            - mpjpe: Mean MPJPE
            - std_mpjpe: Standard deviation
            - num_samples: Number of samples
        """
        results = {}
        for k, v in self.error.items():
            if len(v) > 0:
                results[k] = {
                    "mpjpe": float(np.mean(v)),
                    "std_mpjpe": float(np.std(v)),
                    "num_samples": len(v)
                }
        return results

    def reset(self):
        """Reset accumulated errors."""
        self.error = {'All': []}
        for action in ACTION_NAMES:
            self.error[action] = []

    @abstractmethod
    def eval(self, pred, gt, actions=None, use_action_=False):
        """Evaluate predictions.

        Args:
            pred: Predicted 3D keypoints
            gt: Ground truth 3D keypoints
            actions: Optional list of action names
            use_action_: Whether to track per-action metrics
        """
        raise NotImplementedError

    @abstractmethod
    def desc(self) -> str:
        """Get metric description."""
        raise NotImplementedError


class EvalBody(BaseEval):
    """Evaluate full body MPJPE."""

    def __init__(self):
        super().__init__()

    def eval(self, pred, gt, actions=None, use_action_=False):
        """Evaluate full body predictions.

        Args:
            pred: (N, 16, 3) predicted 3D keypoints
            gt: (N, 16, 3) ground truth 3D keypoints
            actions: (N,) list of action names
            use_action_: Whether to track per-action metrics
        """
        pred = np.array(pred)
        gt = np.array(gt)

        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            gt = gt[np.newaxis, ...]

        # Compute per-sample MPJPE
        errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))  # (N, 16)
        sample_errors = np.mean(errors, axis=-1)  # (N,)

        for i, err in enumerate(sample_errors):
            action = actions[i] if (use_action_ and actions is not None) else None
            self._add_error(float(err), action)

    def desc(self) -> str:
        return "full_body"


class EvalUpperBody(BaseEval):
    """Evaluate upper body MPJPE."""

    def __init__(self):
        super().__init__()
        self.joint_indices = XREGOPOSE_UPPER_BODY_INDICES

    def eval(self, pred, gt, actions=None, use_action_=False):
        """Evaluate upper body predictions.

        Args:
            pred: (N, 16, 3) predicted 3D keypoints
            gt: (N, 16, 3) ground truth 3D keypoints
            actions: (N,) list of action names
            use_action_: Whether to track per-action metrics
        """
        pred = np.array(pred)
        gt = np.array(gt)

        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            gt = gt[np.newaxis, ...]

        # Select upper body joints
        pred_upper = pred[:, self.joint_indices, :]
        gt_upper = gt[:, self.joint_indices, :]

        # Compute per-sample MPJPE
        errors = np.sqrt(np.sum((pred_upper - gt_upper) ** 2, axis=-1))
        sample_errors = np.mean(errors, axis=-1)

        for i, err in enumerate(sample_errors):
            action = actions[i] if (use_action_ and actions is not None) else None
            self._add_error(float(err), action)

    def desc(self) -> str:
        return "upper_body"


class EvalLowerBody(BaseEval):
    """Evaluate lower body MPJPE."""

    def __init__(self):
        super().__init__()
        self.joint_indices = XREGOPOSE_LOWER_BODY_INDICES

    def eval(self, pred, gt, actions=None, use_action_=False):
        """Evaluate lower body predictions.

        Args:
            pred: (N, 16, 3) predicted 3D keypoints
            gt: (N, 16, 3) ground truth 3D keypoints
            actions: (N,) list of action names
            use_action_: Whether to track per-action metrics
        """
        pred = np.array(pred)
        gt = np.array(gt)

        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            gt = gt[np.newaxis, ...]

        # Select lower body joints
        pred_lower = pred[:, self.joint_indices, :]
        gt_lower = gt[:, self.joint_indices, :]

        # Compute per-sample MPJPE
        errors = np.sqrt(np.sum((pred_lower - gt_lower) ** 2, axis=-1))
        sample_errors = np.mean(errors, axis=-1)

        for i, err in enumerate(sample_errors):
            action = actions[i] if (use_action_ and actions is not None) else None
            self._add_error(float(err), action)

    def desc(self) -> str:
        return "lower_body"


def get_action_breakdown_summary(results_dict: dict) -> str:
    """Format results dictionary as a summary string.

    Args:
        results_dict: Results from EvalBody/EvalUpperBody/EvalLowerBody.get_results()

    Returns:
        Formatted string with per-action breakdown
    """
    lines = []

    # Overall first
    if 'All' in results_dict:
        r = results_dict['All']
        lines.append(f"All: {r['mpjpe']:.2f} +/- {r['std_mpjpe']:.2f} mm (n={r['num_samples']})")

    # Per-action
    for action in sorted(ACTION_NAMES):
        if action in results_dict:
            r = results_dict[action]
            lines.append(f"  {action}: {r['mpjpe']:.2f} +/- {r['std_mpjpe']:.2f} mm (n={r['num_samples']})")

    return '\n'.join(lines)
