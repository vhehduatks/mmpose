from mmpose.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F



@MODELS.register_module()
class cross_entropy(nn.Module):

	def __init__(self, loss_weight=1.):
		super().__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.loss_weight = loss_weight

	def forward(self, output, target, target_weight=None):


		loss = self.criterion(output, target)

		return loss * self.loss_weight