from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from os.path import join

import torch
import torchvision.models as models
from torch import nn
import numpy as np

class ResNet50(nn.Module):
	def __init__(self,num_classes=1000):
		super(ResNet50, self).__init__()
		self.num_classes = num_classes
		self.model=models.resnet50()
		self.model.fc=nn.Linear(2048, self.num_classes)
	def forward(self, x):
		x = self.model(x)
		return x







