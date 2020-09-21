import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from .base_detector import BaseDetector
from torchvision import transforms
from PIL import Image
import copy


def cla_decode(output,K):
	dets=torch.topk(output,k=K,dim=1,largest=True)
	return dets

	
def letterbox_image(src,w,h):
	img_h,img_w,_=src.shape
	new_w = img_w
	new_h = img_h
	if((w/img_w) < (h/img_h)): 
		new_w = w
		new_h = (img_h * w)/img_w
	else:
		new_h = h
		new_w = (img_w * h)/img_h
		
	new_w=int(new_w)
	new_h=int(new_h)
	resized =  cv2.resize(src,(int(new_w), int(new_h)))
	boxed = np.full((h,w,3), 0);
	x=int((w-new_w)/2)
	y=int((h-new_h)/2)
	boxed[y:y+new_h,x:x+new_w,:]=resized
	return boxed
	
def crop_image(src):
	img_h,img_w,_=src.shape
	size=img_h
	if(img_h>img_w): 
		size = img_w
	x=int((img_w-size)/2)
	y=int((img_h-size)/2)
	return src[y:y+size,x:x+size,:].copy()
   			   			
	
class ClaDetector(BaseDetector):
	def __init__(self, opt):
		super(ClaDetector, self).__init__(opt)
		self.model.eval()
		self.predict= torch.nn.Softmax(dim=1)
		self.predict.to(self.opt.device)
		
	def process(self, images, return_time=False):
		with torch.no_grad():	
			output = self.model(images)
			output = self.predict(output)
			
			torch.cuda.synchronize()
			forward_time = time.time()
			dets = cla_decode(output=output,K=4)
			output= output.data.cpu().numpy()	
			
		if return_time:
			return output, dets,forward_time
		else:
			return output, dets

	def pre_process(self, src):
		image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)	
		image =	crop_image(image)
		image = cv2.resize(image,(self.opt.input_res,self.opt.input_res))	
		images=image[np.newaxis,:,:,:]
		images = images/255.0
		images = images.astype(np.float32)
		images = torch.from_numpy(images)
		return images,image

	def post_process(self):
		pass

	def merge_outputs(self, detections):
		pass

	def run(self, image_or_path_or_tensor):
		load_time, pre_time, net_time = 0, 0, 0
		tot_time = 0	
		pre_processed = False
		
		start_time = time.time()
		if isinstance(image_or_path_or_tensor, np.ndarray):
			image = image_or_path_or_tensor
		elif type(image_or_path_or_tensor) == type (''): 
			image = cv2.imread(image_or_path_or_tensor)
		else:
			image = image_or_path_or_tensor['image'][0].numpy()
			pre_processed_images = image_or_path_or_tensor
			pre_processed = True	
			
		loaded_time = time.time()
		load_time += (loaded_time - start_time)
		
			
		pre_start_time = time.time()	
		if not pre_processed:
			images,_=self.pre_process(image)		
		images = images.to(self.opt.device).permute((0, 3, 1, 2))
	
		torch.cuda.synchronize()
		pre_process_time = time.time()
		
		pre_time += pre_process_time - pre_start_time		
		output,dets,forward_time=self.process(images=images,return_time=True)
		torch.cuda.synchronize()		
		net_time += forward_time - pre_process_time
		end_time = time.time()
		
		tot_time += end_time - start_time
		return {'output': output,'dets':dets,'tot': tot_time, 'load': load_time,
		'pre': pre_time, 'net':net_time}
		
# def main ():
	# img=cv2.imread('E:/2.png')
	# img=letterbox_image(img,512,512)
	# cv2.imshow('input', img)
	# cv2.waitKey(0)
	
	
# if __name__ == '__main__':
  # main()


		
	