from .dataset.cla import ClaDataset
from .sampler.cla import ClaSampler
import os



def create_label(origin_labels,path):
	f = open(path,"w")
	for label in origin_labels:
		dir=label['fname']
		label=label['label']
		contents="{}|{}\n".format(dir,label)
		f.write(contents)		
	f.close()

def count_category(origin_labels):
	class_to_count={}
	
	for i in range(len(origin_labels)):
		label=origin_labels[i]['label']
		if not label in class_to_count.keys():
			class_to_count[label]=0
			class_to_count[label]+=1
		else:
			class_to_count[label]+=1
			
	def sort_key(x):
		return x[0]
	class_to_count=dict(sorted(class_to_count.items(),key=sort_key,reverse=True))
	return class_to_count
	

def selective_folding(gt_labels,class_name,logger):


	f = open(os.path.join(logger.log_dir,"map.txt"),"w")
	for key in class_name:
		label="{}\n".format(key)
		f.write(label)
	f.close()	

	dataset_count=int(len(gt_labels)*0.2)

	train_labels=gt_labels[dataset_count+dataset_count:]
	valid_labels=gt_labels[dataset_count:dataset_count+dataset_count]
	test_labels=gt_labels[0:dataset_count]
	
	
	create_label(train_labels,os.path.join(logger.log_dir,"train_labels.txt"))
	create_label(valid_labels,os.path.join(logger.log_dir,"valid_labels.txt"))
	create_label(test_labels,os.path.join(logger.log_dir,"test_labels.txt"))
	


	
	train_class_to_count=count_category(train_labels)
	valid_class_to_count=count_category(valid_labels)
	test_class_to_count=count_category(test_labels)

	logger.write("==> data folding:\n")			
	logger.write("train_dataset_to_class_count:{}, sum:{}\n".format(train_class_to_count,len(train_labels)))
	logger.write("valid_dataset_to_class_count:{}, sum:{}\n".format(valid_class_to_count,len(valid_labels)))	
	logger.write("test_dataset_to_class_count:{}, sum:{}\n".format(test_class_to_count,len(test_labels)))
	
	return train_labels,valid_labels,test_labels

	
def read_label(data_dir,anno_dir):
	gt_labels=[]
	with open(anno_dir) as f:
		contents = f.readlines()
		for c in contents:
			c=c.split('\n')[0]
			c=c.split('|')
			fname=os.path.join(data_dir,c[0])
			label=c[1]
			gt_labels.append({'fname': fname,'label': label}) 
	return gt_labels
	
def read_data(data_dir,resume_labels):
	label_map_dir = os.path.join(data_dir, "map.txt")
	label_map=[]
	if os.path.exists(label_map_dir)==True:
		with open(label_map_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split()[0]
				label_map.append(c)	
				
	if resume_labels is False:
		anno_dir = os.path.join(data_dir,"labels.txt")
		gt_labels=read_label(data_dir,anno_dir)
		return gt_labels,label_map
	else:
		anno_dir = os.path.join(data_dir,"train_labels.txt")
		train_labels=read_label(data_dir,anno_dir)
		anno_dir = os.path.join(data_dir,"valid_labels.txt")		
		valid_labels=read_label(data_dir,anno_dir)	
		anno_dir = os.path.join(data_dir,"test_labels.txt")
		test_labels=read_label(data_dir,anno_dir)
		return train_labels,valid_labels,test_labels,label_map
			
	

class Dataset(ClaDataset,ClaSampler):	
	pass
	
def get_dataset():
	return Dataset

