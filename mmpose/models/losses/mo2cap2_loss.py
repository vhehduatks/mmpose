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
	
@MODELS.register_module()
class cosine_similarity(nn.Module):

	def __init__(self, loss_weight=0.01):
		super().__init__()
		self.loss_weight = loss_weight
		self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
	def forward(self, output, target, target_weight=None):

		loss = torch.sum(1 - self.cos(output, target), dim=1)

		return loss * self.loss_weight
	

@MODELS.register_module()
class limb_length(nn.Module):

	def __init__(self, loss_weight=0.5):
		super().__init__()
		self.loss_weight = loss_weight
	def forward(self, output, target, target_weight=None):

		loss = torch.sum(torch.sum(torch.abs(output-target), dim=2), dim=1)

		return loss * self.loss_weight
	
	
@MODELS.register_module()
class heatmap_recon(nn.Module):

	def __init__(self, loss_weight=0.001):
		super().__init__()
		self.loss_weight = loss_weight
	def forward(self, hm_resnet, hm_decoder, target_weight=None):

		loss = torch.sqrt(torch.sum(torch.pow(hm_resnet.reshape(hm_resnet.size(0), -1) - hm_decoder.reshape(hm_decoder.size(0), -1), 2), dim=1))

		return loss * self.loss_weight