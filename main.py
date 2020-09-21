import _init_paths

import os

import torch
import torch.utils.data
import torch.multiprocessing
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from datasets.dataset_factory import get_dataset,read_data,selective_folding
from trains.train_factory import train_factory
from logger import Logger
from datasets.sampler.cla import sampler as ImbalancedDatasetSampler
import numpy as np
import copy



def main(opt):
	torch.manual_seed(opt.seed)
	torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
	
	Dataset = get_dataset()
	Dataset.default_resolution=[512,512]
	if(opt.resume_labels is True):
		train_labels,valid_labels,test_labels,class_name=read_data(opt.data_dir,opt.resume_labels)	
		Dataset.num_classes=len(class_name)
		opt = opts().update_dataset_info_and_set_heads(opt, Dataset)	
		logger = Logger(opt)	
		np.random.shuffle(train_labels)	
	else:
		gt_labels,class_name=read_data(opt.data_dir,opt.resume_labels)
		Dataset.num_classes=len(class_name)
		opt = opts().update_dataset_info_and_set_heads(opt, Dataset)		
		np.random.shuffle(gt_labels)		
		logger = Logger(opt)
		train_labels,valid_labels,test_labels=selective_folding(gt_labels,class_name,logger) 
		
			
	
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
	opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

	print('Creating model...')
	model = create_model(opt.model_name,Dataset.num_classes)
	optimizer = torch.optim.Adam(model.parameters(), opt.lr)
	print(model)
	
	start_epoch = 0
	if opt.load_model=='':
		model, optimizer, start_epoch = load_model(
		  model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)	  
	Trainer = train_factory[opt.task]
	trainer = Trainer(opt, model, optimizer)
	trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)


	print('Setting up data...')
	
	train_set=Dataset(opt,train_labels,class_name)
	train_loader = torch.utils.data.DataLoader(
		train_set, 
		sampler=ImbalancedDatasetSampler(train_set),
		batch_size=int(opt.batch_size/opt.subdivision),
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True,
		drop_last=True
	)
	test_opt=copy.deepcopy(opt)
	test_opt.phase="test"
	valid_set=Dataset(test_opt,valid_labels,class_name)
	valid_loader = torch.utils.data.DataLoader(
		valid_set, 
		batch_size=1,
		shuffle=False,
		num_workers=test_opt.num_workers,
		pin_memory=True,
		drop_last=True
	)
	test_set=Dataset(test_opt,test_labels,class_name)
	test_loader = torch.utils.data.DataLoader(
		test_set, 
		batch_size=1,
		shuffle=False,
		num_workers=test_opt.num_workers,
		pin_memory=True,
		drop_last=True
	)

	
	

	#train
	print('Starting training...')
	
	max_acc_epoch_dir=os.path.join(opt.save_dir, 'model_max_acc.pth')
	if(os.path.exists(max_acc_epoch_dir)):
		checkpoint=torch.load(max_acc_epoch_dir, map_location=lambda storage, loc: storage)
		max_acc_epoch = checkpoint['epoch']
		max_acc=checkpoint['valid_acc']
	else:
		max_acc_epoch=-1
		max_acc=0
	for epoch in range(start_epoch + 1, opt.num_epochs + 1):
		mark = epoch if opt.save_all else 'last'
		log_dict_train, _ = trainer.train(epoch, train_loader)
		log_dict_valid, _ = trainer.test(epoch, valid_loader)
		logger.write('epoch: {} |'.format(epoch))
		logger.write('train loss {:8f} | '.format(log_dict_train['loss']))		
		logger.write('valid loss {:8f} | '.format(log_dict_valid['loss']))
		logger.write('valid acc {:8f} | '.format(log_dict_valid['acc']))
		logger.write('\n')
		if(max_acc < log_dict_valid['acc']):
			max_acc_epoch=epoch
			max_acc=log_dict_valid['acc']
			save_model(max_acc_epoch_dir,epoch, model, optimizer,max_acc)	
			
		save_model(os.path.join(opt.save_dir, 'model_last.pth'),epoch, model, optimizer)
		if epoch in opt.lr_step:
			save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
					 epoch, model, optimizer)
			lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
			print('Drop LR to', lr)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr					
	logger.close()
	
	#test
	print('Final testing...')
	logger.open("log(test).txt")
	model, optimizer, start_epoch = load_model(
	model, max_acc_epoch_dir, optimizer, True, opt.lr, opt.lr_step)	  
	Tester = train_factory[opt.task]
	tester = Tester(opt, model, optimizer)
	tester.set_device(opt.gpus, opt.chunk_sizes, opt.device)
	log_dict_test, _ = tester.test(start_epoch, test_loader)
	logger.write('test model: {}, epoch: {}\n'.format(max_acc_epoch_dir,start_epoch))
	for k, v in log_dict_test.items():
		logger.write('{} {} | '.format(k, v))
	
	


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
	