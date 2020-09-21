from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from os.path import join

import torch
import torchvision.models as models
from torch import nn
import numpy as np

class VGG16(nn.Module):
	def __init__(self,num_classes=1000):
		super(VGG16, self).__init__()
		self.num_classes = num_classes
		self.model=models.vgg16()
		self.model.classifier[6]=nn.Linear(4096, self.num_classes)
		
	def forward(self, x):
		x = self.model(x)
		return x







