# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Mo2Cap2 Evaluation Module

Matches the official MATLAB evaluation protocol (mo2cap2_eval.m):
- Skeleton rescaling to reference bone lengths
- Procrustes alignment WITHOUT scaling (scaling=False)
- Per-action breakdown using official frame ranges
"""
import os
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import scipy.io


# ============================================================================
# Constants
# ============================================================================

MO2CAP2_JOINTS = [
    'Neck', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'
]

MO2CAP2_UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6]  # Neck + Arms
MO2CAP2_LOWER_BODY_INDICES = [7, 8, 9, 10, 11, 12, 13, 14]  # Legs

MO2CAP2_KINEMATIC_PARENTS = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13])

# Official action frame ranges from mo2cap2_eval.m
# Frame numbers are 0-indexed (converted from MATLAB 1-indexed)
ACTION_FRAME_RANGES = {
    'olek_outdoor': {
        'walking':    [(0, 659)],      # 158-817 -> 0-659
        'sitting':    [(859, 1054)],   # 1017-1212 -> 859-1054
        'crawling':   [(2274, 2682)],  # 2432-2840 -> 2274-2682
        'crouching':  [(660, 858)],    # 818-1016 -> 660-858
        'boxing':     [(1311, 1480)],  # 1469-1638 -> 1311-1480
        'dancing':    [(1481, 2025)],  # 1639-2183 -> 1481-2025
        'stretching': [(2026, 2273)],  # 2184-2431 -> 2026-2273
        'waving':     [(1055, 1310)],  # 1213-1468 -> 1055-1310
    },
    'weipeng_studio': {
        'walking':    [(0, 266), (699, 1073), (1480, 1652)],     # 387-653, 1086-1460, 1867-2039
        'sitting':    [(267, 489), (1148, 1479)],                 # 654-876, 1535-1866
        'crawling':   [(490, 698), (2632, 2780)],                 # 877-1085, 3019-3167
        'crouching':  [(2496, 2631)],                             # 2883-3018
        'boxing':     [(1074, 1147), (1653, 1827)],               # 1461-1534, 2040-2214
        'dancing':    [(1828, 2353)],                             # 2215-2740
        'stretching': [(2354, 2495)],                             # 2741-2882
        'waving':     [(2781, 2901)],                             # 3168-3288
    }
}

ACTION_NAMES = ['walking', 'sitting', 'crawling', 'crouching',
                'boxing', 'dancing', 'stretching', 'waving']

__all__ = ["EvalBody", "EvalUpperBody", "EvalLowerBody", "EvalPerJoint"]


# ============================================================================
# Cached data loading
# ============================================================================

@lru_cache(maxsize=1)
def _get_mean3d_path():
    """Get path to mean3D.mat file (cached)."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'utils', 'mean3D.mat'
    )


@lru_cache(maxsize=1)
def _load_mean3d_data():
    """Load mean3D data and precompute bone lengths (cached, loaded once)."""
    mean3D_path = _get_mean3d_path()
    mean3D = scipy.io.loadmat(mean3D_path)['mean3D']  # 3x15 shape
    bones_mean = mean3D - mean3D[:, MO2CAP2_KINEMATIC_PARENTS]
    bone_length = np.sqrt(np.sum(np.power(bones_mean, 2), axis=0))  # 15 shape
    return mean3D, bone_length


# ============================================================================
# Core computation functions (matches MATLAB protocol)
# ============================================================================

def skeleton_rescale(joints, bone_length, kinematic_parents):
    """Rescale skeleton to match reference bone lengths.

    Matches MATLAB: skeleton_rescale(pose, bone_length(2:end), kinematic_parents)

    Args:
        joints: (3, 15) joint positions
        bone_length: (14,) target bone lengths (excluding root)
        kinematic_parents: (15,) parent indices

    Returns:
        (3, 15) rescaled joint positions
    """
    joints_rescaled = np.zeros_like(joints)
    joints_rescaled[:, 0] = joints[:, 0]  # Root stays the same

    for i in range(1, 15):
        parent = kinematic_parents[i]
        bone = joints[:, i] - joints[:, parent]
        bone_norm = np.sqrt(np.sum(bone ** 2))
        if bone_norm > 1e-8:
            bone_rescaled = bone * bone_length[i-1] / bone_norm
        else:
            bone_rescaled = bone
        joints_rescaled[:, i] = joints_rescaled[:, parent] + bone_rescaled

    return joints_rescaled


def procrustes(X, Y, scaling=False, reflection='best'):
    """
    Procrustes analysis - determines linear transformation of Y to best conform to X.

    Matches MATLAB: procrustes(X, Y, 'scaling', false)

    Args:
        X: (N, 3) target coordinates (prediction in MATLAB protocol)
        Y: (N, 3) input coordinates to transform (ground truth in MATLAB protocol)
        scaling: if False, scaling component forced to 1 (MATLAB default: false)
        reflection: 'best', True, or False

    Returns:
        d: residual sum of squared errors
        Z: transformed Y-values (aligned ground truth)
        tform: dict with rotation, scale, translation
    """
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        # MATLAB protocol: scaling=false
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def compute_error_official(pred, gt, joint_sel=None):
    """Compute Mo2Cap2 error following official MATLAB protocol.

    MATLAB protocol:
    1. Rescale both pred and gt to reference bone lengths
    2. Procrustes align gt to pred (NO scaling)
    3. Compute error as pred - aligned_gt

    Args:
        pred: (3, 15) predicted pose
        gt: (3, 15) ground truth pose
        joint_sel: joint selection indices for partial body evaluation

    Returns:
        float: mean joint error
        np.ndarray: per-joint errors (15,)
    """
    _, bone_length = _load_mean3d_data()

    # Rescale to reference bone lengths
    pred_rescale = skeleton_rescale(pred, bone_length[1:], MO2CAP2_KINEMATIC_PARENTS)
    gt_rescale = skeleton_rescale(gt, bone_length[1:], MO2CAP2_KINEMATIC_PARENTS)

    # Procrustes: align gt to pred, NO scaling (matches MATLAB)
    # MATLAB: procrustes(joint3D', pose_gt_3d', 'scaling', false)
    _, gt_aligned, _ = procrustes(pred_rescale.T, gt_rescale.T, scaling=False, reflection='best')
    gt_aligned = gt_aligned.T  # Back to (3, 15)

    # Compute error: pred - aligned_gt (matches MATLAB)
    error = pred_rescale - gt_aligned
    joint_error = np.sqrt(np.sum(error ** 2, axis=0))  # (15,)

    if joint_sel is not None:
        joint_error_sel = joint_error[joint_sel]
        return np.mean(joint_error_sel), joint_error

    return np.mean(joint_error), joint_error


def compute_error_batch_official(pred_batch, gt_batch, joint_sel=None):
    """Batched error computation following official MATLAB protocol.

    Args:
        pred_batch: (B, 15, 3) or (B, 3, 15) predictions
        gt_batch: (B, 15, 3) or (B, 3, 15) ground truth
        joint_sel: joint selection indices for partial body evaluation

    Returns:
        np.ndarray: (B,) mean errors per sample
        np.ndarray: (B, 15) per-joint errors
    """
    # Convert to numpy if tensor
    if hasattr(pred_batch, 'cpu'):
        pred_batch = pred_batch.cpu().numpy()
    elif hasattr(pred_batch, 'numpy'):
        pred_batch = pred_batch.numpy()
    if hasattr(gt_batch, 'cpu'):
        gt_batch = gt_batch.cpu().numpy()
    elif hasattr(gt_batch, 'numpy'):
        gt_batch = gt_batch.numpy()

    B = pred_batch.shape[0]

    # Ensure shape is (B, 3, 15)
    if pred_batch.shape[1] != 3:
        pred_batch = np.transpose(pred_batch, [0, 2, 1])
    if gt_batch.shape[1] != 3:
        gt_batch = np.transpose(gt_batch, [0, 2, 1])

    mean_errors = np.zeros(B)
    joint_errors = np.zeros((B, 15))

    for i in range(B):
        mean_err, joint_err = compute_error_official(pred_batch[i], gt_batch[i], joint_sel)
        mean_errors[i] = mean_err
        joint_errors[i] = joint_err

    return mean_errors, joint_errors


def get_action_from_frame(sequence_name, frame_idx):
    """Get action name from sequence and frame index.

    Args:
        sequence_name: 'olek_outdoor' or 'weipeng_studio'
        frame_idx: 0-indexed frame number within the sequence

    Returns:
        str: action name or 'unknown'
    """
    if sequence_name not in ACTION_FRAME_RANGES:
        return 'unknown'

    for action_name, ranges in ACTION_FRAME_RANGES[sequence_name].items():
        for start, end in ranges:
            if start <= frame_idx <= end:
                return action_name

    return 'unknown'


# ============================================================================
# Base evaluation class
# ============================================================================

class BaseEval(ABC):
    """Base evaluation class for Mo2Cap2 (official protocol)."""

    def __init__(self):
        super().__init__()
        self.error = {'All': []}
        # Initialize all action categories
        for action in ACTION_NAMES:
            self.error[action] = []

    def _add_error(self, err, action=None):
        """Add error to tracking."""
        self.error['All'].append(err)
        if action and action in self.error:
            self.error[action].append(err)

    def get_results(self):
        """Get evaluation results."""
        results = {}
        for k, v in self.error.items():
            if len(v) > 0:
                results[k] = {
                    "mpjpe": float(np.mean(v)),
                    "std_mpjpe": float(np.std(v)),
                    "num_samples": len(v)
                }
        return results

    @abstractmethod
    def eval(self, pred, gt, actions=None, frame_indices=None, sequence_names=None):
        """Evaluate predictions."""
        raise NotImplementedError

    @abstractmethod
    def desc(self):
        """Get metric description."""
        raise NotImplementedError


# ============================================================================
# Evaluation classes
# ============================================================================

class EvalBody(BaseEval):
    """Evaluate full body MPJPE (official protocol)."""

    def __init__(self, mode='mo2cap2', protocol=None):
        super().__init__()

    def eval(self, pred, gt, actions=None, use_action_=False,
             frame_indices=None, sequence_names=None):
        """Evaluate full body predictions.

        Args:
            pred: (B, 15, 3) predictions
            gt: (B, 15, 3) ground truth
            actions: list of action names (optional, legacy support)
            use_action_: whether to track per-action metrics
            frame_indices: list of frame indices for official action lookup
            sequence_names: list of sequence names for official action lookup
        """
        mean_errors, _ = compute_error_batch_official(pred, gt, joint_sel=None)

        for i, err in enumerate(mean_errors):
            action = None
            if use_action_:
                # Try official frame-based action lookup first
                if frame_indices is not None and sequence_names is not None:
                    action = get_action_from_frame(sequence_names[i], frame_indices[i])
                # Fall back to provided action names (sequence-level)
                elif actions and i < len(actions):
                    action = actions[i]

            self._add_error(err, action)

    def desc(self):
        return "FullBody_MPJPE"


class EvalUpperBody(BaseEval):
    """Evaluate upper body MPJPE (Neck + Arms: joints 0-6)."""

    def __init__(self, mode='mo2cap2', protocol=None):
        super().__init__()
        self._joint_sel = MO2CAP2_UPPER_BODY_INDICES

    def eval(self, pred, gt, actions=None, use_action_=False,
             frame_indices=None, sequence_names=None):
        """Evaluate upper body predictions."""
        mean_errors, _ = compute_error_batch_official(pred, gt, joint_sel=self._joint_sel)

        for i, err in enumerate(mean_errors):
            action = None
            if use_action_:
                if frame_indices is not None and sequence_names is not None:
                    action = get_action_from_frame(sequence_names[i], frame_indices[i])
                elif actions and i < len(actions):
                    action = actions[i]

            self._add_error(err, action)

    def desc(self):
        return "UpperBody_MPJPE"


class EvalLowerBody(BaseEval):
    """Evaluate lower body MPJPE (Legs: joints 7-14)."""

    def __init__(self, mode='mo2cap2', protocol=None):
        super().__init__()
        self._joint_sel = MO2CAP2_LOWER_BODY_INDICES

    def eval(self, pred, gt, actions=None, use_action_=False,
             frame_indices=None, sequence_names=None):
        """Evaluate lower body predictions."""
        mean_errors, _ = compute_error_batch_official(pred, gt, joint_sel=self._joint_sel)

        for i, err in enumerate(mean_errors):
            action = None
            if use_action_:
                if frame_indices is not None and sequence_names is not None:
                    action = get_action_from_frame(sequence_names[i], frame_indices[i])
                elif actions and i < len(actions):
                    action = actions[i]

            self._add_error(err, action)

    def desc(self):
        return "LowerBody_MPJPE"


class EvalPerJoint(object):
    """Evaluate per-joint MPJPE (official protocol)."""

    def __init__(self, mode='mo2cap2', protocol=None):
        super().__init__()
        self.joint_errors = []  # (N, 15)

    def eval(self, pred, gt):
        """Evaluate per-joint errors.

        Args:
            pred: (B, 15, 3) predictions
            gt: (B, 15, 3) ground truth
        """
        _, joint_errors = compute_error_batch_official(pred, gt, joint_sel=None)
        self.joint_errors.extend(joint_errors.tolist())

    def get_results(self):
        """Get mean per-joint errors."""
        if len(self.joint_errors) == 0:
            return np.zeros(15)
        stacked = np.array(self.joint_errors)
        return np.mean(stacked, axis=0)


# ============================================================================
# Utility functions
# ============================================================================

def get_action_breakdown_summary(results_dict):
    """Format action breakdown for logging.

    Args:
        results_dict: dict from get_results()

    Returns:
        str: formatted summary
    """
    lines = []

    # Overall
    if 'All' in results_dict:
        lines.append(f"Overall: {results_dict['All']['mpjpe']:.2f}mm "
                    f"(std: {results_dict['All']['std_mpjpe']:.2f}, "
                    f"n={results_dict['All']['num_samples']})")

    # Per-action
    lines.append("Per-action breakdown:")
    for action in ACTION_NAMES:
        if action in results_dict:
            r = results_dict[action]
            lines.append(f"  {action:12s}: {r['mpjpe']:7.2f}mm (n={r['num_samples']})")

    return '\n'.join(lines)
