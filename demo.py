from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import torch
import numpy as np


from opts import opts
from detectors.detector_factory import detector_factory
from models.model import create_model, load_model
import utils.interp_utils as interp_utils
import time





image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
result_stats=['dets','output']

def read_map(data_dir):
	label_map_dir = os.path.join(data_dir, "map.txt")
	label_map=[]
	if os.path.exists(label_map_dir)==True:
		with open(label_map_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split()[0]
				label_map.append(c)
				
	num_classes=len(label_map)
	ind_to_class = dict(zip(range(num_classes),label_map))
	return ind_to_class,num_classes
	

def letterbox_image(src,w,h):
	img_h,img_w,_=src.shape
	new_w = img_w
	new_h = img_h
	if((w/img_w) < (h/img_h)): 
		new_w = w;
		new_h = (img_h * w)/img_w
	else:
		new_h = h;
		new_w = (img_w * h)/img_h
	new_w=int(new_w)
	new_h=int(new_h)
	resized =  cv2.resize(src,(int(new_w), int(new_h)))
	boxed = np.full((h,w,3), 0);
	x=int((w-new_w)/2)
	y=int((h-new_h)/2)
	boxed[y:y+new_h,x:x+new_w,:]=resized
	return boxed







def demo(opt):
	path='./export/'
	arch='ResNet50d'
	ind_to_class,num_classes=read_map(path)
	opt.load_model=os.path.join(path,'model.pth')
	opt.arch=arch
	# ind_to_class=read_map('H:/KAERI/Custom/data/custom/dataset')	
	opt.num_classes=num_classes
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
	Detector = detector_factory[opt.task]
	detector = Detector(opt)  
	interp_utils.init(detector.model)
	
	times = []	
	if opt.demo == 'webcam' or \
		opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
		cam = cv2.VideoCapture((0 if opt.demo == 'webcam' else opt.demo) +cv2.CAP_DSHOW)
		cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)			
		cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
	
	
		detector.pause = False
		while True:
			_, img = cam.read()
			default_resolution=opt.default_resolution
			input,raw_image=detector.pre_process(img) 
			interp_rets=interp_utils.interpreter(input,raw_image)
			ret = detector.run(img)
			img=cv2.cvtColor(raw_image.astype('uint8'), cv2.COLOR_RGB2BGR)
			
			
			time_str = ''
			for stat in time_stats:
				if stat in ret:
					time_str = time_str + '{} {:8f}s |'.format(stat, ret[stat])
					
			result_str=''
			for stat in result_stats:
				if stat in ret:
					if(stat=='output'):
						predict=ret[stat]
						probs=predict.tolist()[0]
						top_n=(-predict).argsort()[0]
						top_n=top_n[:3]
						top_n=top_n.tolist()
						result=[]
						for i in top_n:
							result.append("{}({:2f})".format(ind_to_class[i],probs[i]))
						result_str = result_str + '{} {}|'.format(stat, result)
						
						top_1=top_n[0]
						img = cv2.putText( np.asarray(img), "top {}: {}({})".format(1,ind_to_class[top_1],probs[top_1]), (0, 30),
						cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)	

						rest_top_n=top_n[1:]
						for i,p in enumerate(rest_top_n):
							img = cv2.putText( np.asarray(img), "top {}: {}({})".format(i+2,ind_to_class[p],probs[p]), (0, 30+(i+1)*30),
							cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)				
			tot_str=''
			tot_str=tot_str+time_str+result_str
			cv2.imshow('prediction_top_n', img)
			for i,rets in enumerate(interp_rets):
				cv2.imshow('top_{}_guide_cam'.format(i+1), interp_rets[i]["guide_cam"])
				cv2.imshow('top_{}_grad_cam'.format(i+1), interp_rets[i]["grad_cam"])
				cv2.imshow('top_{}_guided_grad_cam'.format(i+1), interp_rets[i]["guided_grad_cam"])
			
			if cv2.waitKey(1) == 27:
				return  
	else:
		if os.path.isdir(opt.demo):
			image_names = []
			ls = os.listdir(opt.demo)
			for file_name in sorted(ls):
			  ext = file_name[file_name.rfind('.') + 1:].lower()
			  if ext in image_ext:
				  image_names.append(os.path.join(opt.demo, file_name))
		else:
			image_names = [opt.demo]
		
		for (image_name) in image_names:
			ret = detector.run(image_name)
			time_str = ''
			for stat in time_stats:
				if stat in ret:
					time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
			print(time_str)
			
	interp_utils.release()
	
if __name__ == '__main__':
  opt = opts().init()  
  opt.default_resolution=[512,512] 
  demo(opt)
