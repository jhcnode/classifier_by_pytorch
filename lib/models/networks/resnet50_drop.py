from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from os.path import join

import torch
import torchvision.models as models
from torch import nn
import numpy as np

class ResNet50d(nn.Module):
	def __init__(self,num_classes=1000):
		super(ResNet50d, self).__init__()
		self.num_classes = num_classes
		self.model=models.resnet50()
		self.model.fc=nn.Sequential(
		nn.Dropout(0.5),			
		nn.Linear(2048, self.num_classes) 
		)
	def forward(self, x):
		x = self.model(x)
		return x







