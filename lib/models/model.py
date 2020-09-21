from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import os
from .networks.vgg16 import VGG16
from .networks.resnet18 import ResNet18
from .networks.resnet50 import ResNet50
from .networks.resnext50 import ResNext50
from .networks.wide_resnet50 import WideResNet50
from .networks.resnet50_drop import ResNet50d
_model_factory = {
	'vgg16': VGG16,
	'resnet18': ResNet18,
	'resnet50d': ResNet50d,
	'resnet50': ResNet50,
	'resnet50': ResNext50,
	'wideresnet50': WideResNet50	
	
}

def create_model(model_name,num_classes=1000):
	model_name=model_name.lower()
	get_model = _model_factory[model_name]
	model = get_model(num_classes=num_classes)
	return model

def load_model(model, model_path, optimizer=None, resume=False, 
				lr=None, lr_step=None):
	start_epoch = 0
	checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
	print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
	state_dict_ = checkpoint['state_dict']
	state_dict = {}

	# convert data_parallal to model
	for k in state_dict_:
		if k.startswith('module') and not k.startswith('module_list'):
			state_dict[k[7:]] = state_dict_[k]
		else:
			state_dict[k] = state_dict_[k]
		model_state_dict = model.state_dict()

	# check loaded parameters and created model parameters
	for k in state_dict:
		if k in model_state_dict:
			if state_dict[k].shape != model_state_dict[k].shape:
				print('Skip loading parameter {}, required shape{}, '\
				'loaded shape{}.'.format(
				k, model_state_dict[k].shape, state_dict[k].shape))
				state_dict[k] = model_state_dict[k]
		else:
			print('Drop parameter {}.'.format(k))
			
			
	for k in model_state_dict:
		if not (k in state_dict):
			print('No param {}.'.format(k))
			state_dict[k] = model_state_dict[k]
		model.load_state_dict(state_dict, strict=False)

	# resume optimizer parameters
	if optimizer is not None and resume is True:
		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			start_lr = lr
			for step in lr_step:
				if start_epoch >= step:
					start_lr *= 0.1
				
			for param_group in optimizer.param_groups:
				param_group['lr'] = start_lr
			print('Resumed optimizer with start lr', start_lr)
		else:
			print('No optimizer parameters in checkpoint.')
		  
	
	if optimizer is not None:
		return model, optimizer, start_epoch
	else:
		return model

def save_model(path, epoch, model, optimizer=None, valid_acc=None):
	if isinstance(model, torch.nn.DataParallel):
		state_dict = model.module.state_dict()
	else:
		state_dict = model.state_dict()
	data = {'epoch': epoch,
          'state_dict': state_dict,
          'valid_acc': valid_acc}
	if not (optimizer is None):
		data['optimizer'] = optimizer.state_dict()
	torch.save(data, path)

