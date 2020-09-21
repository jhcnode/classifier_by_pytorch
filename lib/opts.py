from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()

	
    self.parser.add_argument('--task', default='cla',
                             help='cla')
    self.parser.add_argument('--dataset', default='custom',
                             help='custom')					
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--demo', default='webcam', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')	
    self.parser.add_argument('--data_dir', default=None,
                             help='data directory')								  
    self.parser.add_argument('--resume_labels', type=bool,default=False,
                             help='resume_labels')								  
								  								  
    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') 
	#log				
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')	
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
							 
    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--subdivision', type=int, default=16,
                             help='subdivision')							 
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
							 
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 		

	#test
    self.parser.add_argument('--test_dir', default='', 
                             help='test_dir')

							 
    # model
    self.parser.add_argument('--model_name', default='ResNet18', 
                             help='model architecture. Currently tested')		
    #demo
    self.parser.add_argument('--cam_res', default='1920,1080', 
                             help='delimited list input format:[w,h] '
                                  'ex) --cam_res="w,h"')		
    self.parser.add_argument('--cam_focus',type=int, default=0, 
                             help='camera focus')		


  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.phase="train"
    opt.task=opt.task
    opt.subdivision=opt.subdivision
    opt.resume_labels=opt.resume_labels
    opt.cam_focus=int(opt.cam_focus)
    opt.cam_res=[int(item)for item in opt.cam_res.split(',')]
	
    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.root_dir= os.path.abspath(opt.root_dir)
    if(opt.data_dir is None):
      opt.data_dir = os.path.join( os.path.abspath(opt.root_dir), 'data')
		
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)


    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.num_classes = dataset.num_classes
	
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.input_res = max(opt.input_h, opt.input_w)
    if opt.load_model == '' and opt.resume is False:
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')

    return opt

  def init(self, args=''):
    default_dataset_info = {
    'cla': {'default_resolution': [512, 512], 'dataset': 'custom','num_classes': 4}
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
	
    
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
