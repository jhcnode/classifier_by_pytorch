import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import random
from torchvision import transforms

class RandomRotationCase:
	def __init__(self, angles):
		self.angles = angles

	def __call__(self, x):
		angle = random.choice(self.angles)
		return TF.rotate(x, angle)
		
class RandomNoise(object):
	def __init__(self, probability):
		self.probability = probability
		 
	def noise(self,x,prob):
		x=x.numpy()
		rnd = np.random.rand(x.shape[0],x.shape[1],x.shape[2])
		noisy = x[:]
		noisy[rnd < prob] = np.random.rand(1)
		return torch.from_numpy(noisy)
		
	def __call__(self, x):
		if random.random() <= self.probability:
			return self.noise(x,prob=0.05)
		return x		


class ClaDataset(data.Dataset):
	default_resolution=[512,512]
	num_classes=-1
	def __init__(self,opt,gt_labels,class_name):
		super(ClaDataset, self).__init__()
		self.opt=opt
		self.gt_labels = gt_labels
		self.class_name = class_name
		self.num_classes=len(self.class_name)
		self.class_to_ind = dict(zip(self.class_name, range(self.num_classes)))
		self.num_samples=len(self.gt_labels)
		self.transform1 = transforms.Compose([
												transforms.RandomResizedCrop(size=(ClaDataset.default_resolution[0],ClaDataset.default_resolution[1]),scale=(0.7, 1.0)),
												transforms.ColorJitter(brightness=0.75,contrast=0.4,saturation=0.75,hue=0.1),
												transforms.RandomVerticalFlip(),
												transforms.RandomHorizontalFlip(),
												RandomRotationCase([0,90,180,270]),
												transforms.RandomRotation(60),
												transforms.ToTensor(),
												RandomNoise(0.5)
                                           ])
				
		self.transform2 = transforms.Compose([	
												transforms.Resize(size=(ClaDataset.default_resolution[0],ClaDataset.default_resolution[1])),
												transforms.ColorJitter(brightness=0.75,contrast=0.4,saturation=0.75,hue=0.1),
												transforms.RandomVerticalFlip(),
												transforms.RandomHorizontalFlip(),
												RandomRotationCase([0,90,180,270]),
												transforms.RandomRotation(60),
												transforms.ToTensor(),
												RandomNoise(0.5)
											])
		self.test_transform = transforms.Compose([transforms.Resize(size=(ClaDataset.default_resolution[0],ClaDataset.default_resolution[1])),
												transforms.ToTensor()])
		
	def __len__(self):
		 return self.num_samples
