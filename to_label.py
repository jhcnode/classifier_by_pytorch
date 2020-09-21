import os
import glob
import argparse
import cv2
import numpy as np
import json
from PIL import Image

import shutil




def to_label(path):
	labels={}
	class_to_ind={}
	idx=0
	itr = iter(os.walk(path))
	next(itr) 
	for dirpath, _, filenames in itr:
		class_to_ind[os.path.basename(dirpath)]=idx
		idx+=1
	for dirpath, _, filenames in os.walk(path):
		for filename in filenames:	
			labels[os.path.join(dirpath,filename).replace('\\','/')]=os.path.basename(dirpath)
			
	f = open(os.path.join(path,"labels.txt"),"w")
		
	for dir,label in labels.items():
		label="{}|{}\n".format(dir,label)
		f.write(label)			
	f.close()

	f = open(os.path.join(path,"map.txt"),"w")
	for key,val in class_to_ind.items():
		label="{}\n".format(key)
		f.write(label)
	f.close()	
	
	

				

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d","--dir", default="E:\kaeri\kaeri")
	args = parser.parse_args()
	dir_path=os.path.join(args.dir)
	to_label(dir_path)
	
	
	
	



main()
