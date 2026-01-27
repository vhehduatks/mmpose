from mmpose.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
pred, gt shape b,15,3
'''

@MODELS.register_module()
class pose_l2norm(nn.Module):

	def __init__(self, loss_weight=1.):
		super().__init__()
		self.loss_weight = loss_weight

	def forward(self, output, target, target_weight=None):


		loss = torch.sqrt(torch.sum(torch.sum(torch.pow(output-target, 2), dim=2), dim=1))

		return loss * self.loss_weight
	
# XR-EgoPose skeleton: (parent_idx, child_idx) pairs for 15 limbs
# Keypoints: 0=Spine2, 1=Head, 2=LeftArm, 3=LeftForeArm, 4=LeftHand,
#            5=RightArm, 6=RightForeArm, 7=RightHand, 8=LeftUpLeg,
#            9=LeftLeg, 10=LeftFoot, 11=LeftToeBase, 12=RightUpLeg,
#            13=RightLeg, 14=RightFoot, 15=RightToeBase
EGOPOSE_SKELETON = [
	(0, 1),   # Spine2 -> Head
	(0, 2),   # Spine2 -> LeftArm
	(2, 3),   # LeftArm -> LeftForeArm
	(3, 4),   # LeftForeArm -> LeftHand
	(0, 5),   # Spine2 -> RightArm
	(5, 6),   # RightArm -> RightForeArm
	(6, 7),   # RightForeArm -> RightHand
	(0, 8),   # Spine2 -> LeftUpLeg
	(8, 9),   # LeftUpLeg -> LeftLeg
	(9, 10),  # LeftLeg -> LeftFoot
	(10, 11), # LeftFoot -> LeftToeBase
	(0, 12),  # Spine2 -> RightUpLeg
	(12, 13), # RightUpLeg -> RightLeg
	(13, 14), # RightLeg -> RightFoot
	(14, 15), # RightFoot -> RightToeBase
]


@MODELS.register_module()
class cosine_similarity(nn.Module):
	"""Limb direction cosine similarity loss (L_cos in paper).

	Computes cosine similarity between predicted and GT limb direction vectors.
	This is semantically correct as it compares bone orientations rather than
	raw 3D coordinates.

	Default loss_weight: λ_cos = 0.1 (Section IV-A)

	Args:
		loss_weight (float): Weight of the loss. Default: 0.1
		skeleton (list): List of (parent_idx, child_idx) tuples defining limbs.
			Default: EGOPOSE_SKELETON
	"""

	def __init__(self, loss_weight=0.1, skeleton=None):
		super().__init__()
		self.loss_weight = loss_weight
		self.skeleton = skeleton if skeleton is not None else EGOPOSE_SKELETON
		self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

	def forward(self, output, target, target_weight=None):
		"""
		Args:
			output: Predicted 3D keypoints (batch, num_joints, 3)
			target: GT 3D keypoints (batch, num_joints, 3)
			target_weight: Optional joint weights (not used)

		Returns:
			loss: (batch,) tensor of cosine similarity losses
		"""
		batch_size = output.shape[0]
		total_loss = torch.zeros(batch_size, device=output.device)

		for parent_idx, child_idx in self.skeleton:
			# Compute limb direction vectors
			pred_limb = output[:, child_idx, :] - output[:, parent_idx, :]
			gt_limb = target[:, child_idx, :] - target[:, parent_idx, :]

			# Cosine similarity loss: 1 - cos(pred_limb, gt_limb)
			cos_sim = self.cos(pred_limb, gt_limb)
			total_loss = total_loss + (1.0 - cos_sim)

		return total_loss * self.loss_weight


@MODELS.register_module()
class limb_length(nn.Module):
	"""L1 norm loss (L_limbL1 in paper) - computes L1 distance between poses.

	Default loss_weight: λ_L1 = 0.25 (Section IV-A)
	"""

	def __init__(self, loss_weight=0.25):
		super().__init__()
		self.loss_weight = loss_weight

	def forward(self, output, target, target_weight=None):
		loss = torch.sum(torch.sum(torch.abs(output-target), dim=2), dim=1)
		return loss * self.loss_weight
	
	
# Left-Right symmetric limb pairs for symmetry loss
# Each tuple: ((parent_L, child_L), (parent_R, child_R))
EGOPOSE_SYMMETRIC_LIMBS = [
	((0, 2), (0, 5)),     # Spine2→LeftArm, Spine2→RightArm
	((2, 3), (5, 6)),     # LeftArm→LeftForeArm, RightArm→RightForeArm
	((3, 4), (6, 7)),     # LeftForeArm→LeftHand, RightForeArm→RightHand
	((0, 8), (0, 12)),    # Spine2→LeftUpLeg, Spine2→RightUpLeg
	((8, 9), (12, 13)),   # LeftUpLeg→LeftLeg, RightUpLeg→RightLeg
	((9, 10), (13, 14)),  # LeftLeg→LeftFoot, RightLeg→RightFoot
	((10, 11), (14, 15)), # LeftFoot→LeftToeBase, RightFoot→RightToeBase
]


@MODELS.register_module()
class bone_length_loss(nn.Module):
	"""Bone length constraint loss.

	Enforces predicted bone lengths to match GT bone lengths.
	Unlike limb_length (generic L1), this computes actual per-limb
	Euclidean bone length and penalizes length mismatch.

	This constrains lower body joints (hip→knee→ankle→foot) to
	maintain realistic limb proportions, especially in the depth axis.

	Args:
		loss_weight (float): Weight of the loss. Default: 0.5
		skeleton (list): List of (parent_idx, child_idx) tuples.
	"""

	def __init__(self, loss_weight=0.5, skeleton=None):
		super().__init__()
		self.loss_weight = loss_weight
		self.skeleton = skeleton if skeleton is not None else EGOPOSE_SKELETON

	def forward(self, output, target, target_weight=None):
		batch_size = output.shape[0]
		total_loss = torch.zeros(batch_size, device=output.device)

		for parent_idx, child_idx in self.skeleton:
			pred_len = torch.norm(
				output[:, child_idx] - output[:, parent_idx], dim=-1
			)
			gt_len = torch.norm(
				target[:, child_idx] - target[:, parent_idx], dim=-1
			)
			total_loss = total_loss + F.smooth_l1_loss(
				pred_len, gt_len, reduction='none'
			)

		return total_loss * self.loss_weight


@MODELS.register_module()
class pose_l2norm_weighted(nn.Module):
	"""Body-part weighted L2 norm loss.

	Applies higher loss weight to lower body joints (indices 8-15)
	to compensate for HMD signal absence in lower body.

	Args:
		loss_weight (float): Overall loss weight. Default: 1.0
		lower_body_weight (float): Multiplier for lower body joints. Default: 1.5
	"""

	def __init__(self, loss_weight=1., lower_body_weight=1.5):
		super().__init__()
		self.loss_weight = loss_weight
		self.lower_body_weight = lower_body_weight

	def forward(self, output, target, target_weight=None):
		# Per-joint L2 error: (B, 16)
		per_joint = torch.sqrt(
			torch.sum(torch.pow(output - target, 2), dim=2)
		)
		# Build joint weights on same device
		joint_weights = torch.ones(
			per_joint.shape[1], device=output.device
		)
		joint_weights[8:] = self.lower_body_weight
		# Weighted sum
		loss = (per_joint * joint_weights).sum(dim=1)
		return loss * self.loss_weight


@MODELS.register_module()
class symmetry_loss(nn.Module):
	"""Left-right bone length symmetry loss.

	Enforces symmetric limbs (left arm/right arm, left leg/right leg)
	to have equal bone lengths in predictions.

	Args:
		loss_weight (float): Weight of the loss. Default: 0.1
		symmetric_limbs (list): List of ((pL, cL), (pR, cR)) tuples.
	"""

	def __init__(self, loss_weight=0.1, symmetric_limbs=None):
		super().__init__()
		self.loss_weight = loss_weight
		self.symmetric_limbs = (
			symmetric_limbs if symmetric_limbs is not None
			else EGOPOSE_SYMMETRIC_LIMBS
		)

	def forward(self, output, target=None, target_weight=None):
		batch_size = output.shape[0]
		total_loss = torch.zeros(batch_size, device=output.device)

		for (pl, cl), (pr, cr) in self.symmetric_limbs:
			left_len = torch.norm(
				output[:, cl] - output[:, pl], dim=-1
			)
			right_len = torch.norm(
				output[:, cr] - output[:, pr], dim=-1
			)
			total_loss = total_loss + torch.abs(left_len - right_len)

		return total_loss * self.loss_weight


@MODELS.register_module()
class heatmap_recon(nn.Module):

	def __init__(self, loss_weight=0.001):
		super().__init__()
		self.loss_weight = loss_weight
	def forward(self, hm_resnet, hm_decoder, target_weight=None):

		loss = torch.sqrt(torch.sum(torch.pow(hm_resnet.reshape(hm_resnet.size(0), -1) - hm_decoder.reshape(hm_decoder.size(0), -1), 2), dim=1))

		return loss * self.loss_weight