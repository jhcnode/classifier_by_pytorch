import torch.utils.data as data
import numpy as np
import torch
from PIL import Image


def sampler(train_set):
	
	labels=[]
	for i in range(len(train_set.gt_labels)):
		labels.append(train_set.class_to_ind[train_set.gt_labels[i]['label']])
		
	labels=np.array(labels)
	_, counts = np.unique(labels, return_counts=True)

	weights = 1.0 / torch.tensor(counts, dtype=torch.float)
	sample_weights = weights[labels]
	sampler =  torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
	return sampler

class ImbalancedDatasetSampler(data.sampler.Sampler):
	def __init__(self, dataset, indices=None, num_samples=None):	
		self.indices = list(range(len(dataset))) if indices is None else indices
		self.num_samples = len(self.indices) if num_samples is None else num_samples	
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset.gt_labels[idx]['label']
				
	def __iter__(self):
		items=(self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
		return items

	def __len__(self):
		return self.num_samples


class ClaSampler(data.Dataset):
	def __getitem__(self, index):
		img_path=self.gt_labels[index]['fname']
		class_name=self.gt_labels[index]['label']

		try:
			img = Image.open(img_path)
			width, height = img.size
			if(self.opt.phase=="train"):
				img = self.transform1(img)
			else:
				img = self.test_transform(img)	
		except Exception as err:
			print("error image: {}".format(img_path))
			return
		label = self.class_to_ind[class_name]
		ret = {'input': img,'label': label}		
		return ret
		
		