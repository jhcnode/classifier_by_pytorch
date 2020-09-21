
import torch
import numpy as np

from .base_trainer import BaseTrainer

class CLALoss(torch.nn.Module):
	def __init__(self, opt):
		super(CLALoss, self).__init__()
		self.crit = torch.nn.CrossEntropyLoss()
		self.opt = opt

	def forward(self, outputs, batch):
		_batch=batch['label'].long()
		loss = self.crit(outputs, _batch)
		loss_stats = {'loss': loss}
		return loss, loss_stats

class CLATrainer(BaseTrainer):
	def __init__(self, opt, model, optimizer=None):
		super(CLATrainer, self).__init__(opt, model, optimizer=optimizer)
		
	def _get_losses(self, opt):
		loss_states = ['loss']
		loss = CLALoss(opt)
		return loss_states, loss
		
	def _get_evals(self, opt):
		eval_states = ['acc']	
		return eval_states
		
	def _get_result(self,batch,output):	
		_, predicted = torch.max(output.data, 1)
		total =batch['label'].size(0)
		correct = (predicted == batch['label']).sum().item()
		return {'acc':correct/total}


		