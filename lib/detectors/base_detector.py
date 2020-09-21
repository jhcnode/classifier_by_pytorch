import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model


class BaseDetector(object):
	def __init__(self, opt):
		if opt.gpus[0] >= 0:
		  opt.device = torch.device('cuda')
		else:
		  opt.device = torch.device('cpu')

		print('Creating model...')
		self.model = create_model(opt.arch, opt.num_classes)
		self.model = load_model(self.model, opt.load_model)
		self.model = self.model.to(opt.device)
		
		self.num_classes = opt.num_classes
		self.opt = opt
		self.pause = True

	def pre_process(self, image):
		raise NotImplementedError

	def process(self, images, return_time=False):
		raise NotImplementedError

	def post_process(self):
		raise NotImplementedError

	def merge_outputs(self, detections):
		raise NotImplementedError

	def run(self, image_or_path_or_tensor):
		raise NotImplementedError


