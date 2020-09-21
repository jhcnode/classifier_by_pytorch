
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap,QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout,QAction
from PyQt5.QtCore import Qt
import threading


import _init_paths
import os
import cv2
import torch
import numpy as np
from opts import opts
from detectors.detector_factory import detector_factory
from models.model import create_model, load_model
import time

video_ext = ['mp4', 'mov', 'avi', 'mkv']



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

def zoom_io(img,zoom_factor):
	h,w,c=img.shape
	
	cy=int(h/2)
	cx=int(w/2)
	
	
	def lerp(p1,p2,d):
		return (1-d)*p1 + d*p2
		
	lower_bound=0.1
	w_lower_bound=lower_bound*w
	h_lower_bound=lower_bound*h
	
	w=int(lerp(w_lower_bound,w,1-zoom_factor))
	h=int(lerp(h_lower_bound,h,1-zoom_factor))
	cx=cx-int(w/2)
	cy=cy-int(h/2)
	img=img[cy:cy+h,cx:cx+w]
	return img

	
def run(opt,demo_gui):
	path='./export/'
	arch='ResNet50d'
	ind_to_class,num_classes=read_map(path)	
	opt.load_model=os.path.join(path,'model.pth')
	opt.arch=arch
	opt.num_classes=num_classes
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
	Detector = detector_factory[opt.task]
	detector = Detector(opt)  
	times = []	
	if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
		cam = cv2.VideoCapture((0 if opt.demo == 'webcam' else opt.demo) +cv2.CAP_DSHOW)
		cam.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.cam_res[1])			
		cam.set(cv2.CAP_PROP_FRAME_WIDTH, opt.cam_res[0])	
		cam.set(cv2.CAP_PROP_AUTOFOCUS, opt.cam_focus) 
		demo_gui.gui['imageView'].resize(demo_gui.title_width, demo_gui.title_height)
		detector.pause = False
		while demo_gui.is_run:
			_, img = cam.read()
			img=zoom_io(img,demo_gui.zoom_factor)
			input,raw_image=detector.pre_process(img) 
			ret = detector.run(img)
			# print(ret)
			demo_gui.update_gui(ret,ind_to_class)
			img=raw_image
			img = cv2.resize(img, (demo_gui.title_width, demo_gui.title_height))
			h,w,c = img.shape	
			qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
			pixmap = QtGui.QPixmap.fromImage(qImg)
			demo_gui.gui['imageView'].setPixmap(pixmap)
		cam.release()
	print("close")
	
class Demo(QWidget):
	def __init__(self,opt):
		super().__init__()
		self.opt=opt
		self.zoom_factor=0
		self.title_width=768
		self.title_height=768
		self.offsetx=128
		self.offsety=256
		self.offsetx1=-70
		self.gui= {
		"top1":QLabel(self),
		"top2":QLabel(self),
		"top3":QLabel(self),
		"top1_val":QLabel(self),
		"top2_val":QLabel(self),
		"top3_val":QLabel(self),
		"imageView":QLabel()
		} 

		self.is_run=True
		self.th = threading.Thread(target=run,args=(self.opt,self))

		
		self.vbox = QVBoxLayout(self)
		self.initUI()
		
	
	def initUI(self):
		finish = QAction("Quit", self)
		finish.triggered.connect(self.closeEvent)
	
		self.setStyleSheet("background-color:white;")
		self.vbox.setContentsMargins(0,0,0,0)
		self.vbox.addWidget(self.gui['imageView'])
		self.vbox.setAlignment(Qt.AlignTop)
		
		
		#top1
		self.gui['top1'].setStyleSheet(
		"""
		font-family:  SpoqaHanSans-Bold;
		font-size: 24px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #000000;
		"""
		)
		self.gui['top1'].setAlignment(Qt.AlignLeft)
		self.gui['top1'].move(37,564+self.offsety)
		self.gui['top1'].setText("None")
		self.gui['top1'].adjustSize()
		
		
		# top2
		self.gui['top2'].setStyleSheet("""
		font-family:  SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #151515;
		"""
		)
		self.gui['top2'].setAlignment(Qt.AlignLeft)
		self.gui['top2'].move(self.title_width-175+self.offsetx1,545+self.offsety)
		self.gui['top2'].setText("None")
		self.gui['top2'].adjustSize()
		
		
		# top3
		self.gui['top3'].setStyleSheet("""
		font-family:  SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #151515;
		"""
		)
		self.gui['top3'].setAlignment(Qt.AlignLeft)
		self.gui['top3'].move(self.title_width-175+self.offsetx1,578+self.offsety)
		self.gui['top3'].setText("None")
		self.gui['top3'].adjustSize()
		
		# top1_val
		self.gui['top1_val'].setStyleSheet("""
		width: 70px;
		height: 26px;
		border-radius: 6px;
		background-color: #2274ff;
		
		font-family: SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #ffffff;	
		"""
		)
		self.gui['top1_val'].setAlignment(Qt.AlignCenter)
		self.gui['top1_val'].move(132,572+self.offsety)
		self.gui['top1_val'].setText("00.00%")
		
		
		# top2_val
		self.gui['top2_val'].setStyleSheet("""
		width: 57px;
		height: 21px;
		border-radius: 4.9px;
		background-color: #666666;
		
		font-family: SpoqaHanSans-Bold;
		font-size: 11px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #ffffff;	
		"""
		)
		self.gui['top2_val'].setAlignment(Qt.AlignCenter)
		self.gui['top2_val'].move(self.title_width-108+self.offsetx1,545+self.offsety)
		self.gui['top2_val'].setText("00.00%")		
		
		# top3_val
		self.gui['top3_val'].setStyleSheet("""
		width: 57px;
		height: 21px;
		border-radius: 4.9px;
		background-color: #bcbcbc;
		
		font-family: SpoqaHanSans-Bold;
		font-size: 11px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #ffffff;	
		"""
		)
		self.gui['top3_val'].setAlignment(Qt.AlignCenter)
		self.gui['top3_val'].move(self.title_width-108+self.offsetx1,578+self.offsety)
		self.gui['top3_val'].setText("00.00%")		
		
				
						
		
		
		
		
		#textview
		top1_textView= QLabel(self)
		top1_textView.setStyleSheet("""
		font-family: SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #909090;
		"""
		)
		
		top1_textView.setAlignment(Qt.AlignLeft)
		top1_textView.setText("TOP 01.")
		top1_textView.move(37,545+self.offsety)
		top1_textView.adjustSize()
		
		
		top2_textView= QLabel(self)
		top2_textView.setStyleSheet("""
		font-family: SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #909090;
		"""
		)
		top2_textView.setAlignment(Qt.AlignLeft)
		top2_textView.setText("TOP 02.")
		top2_textView.move(self.title_width-238+self.offsetx1,545+self.offsety)
		top2_textView.adjustSize()
		
		top3_textView= QLabel(self)
		top3_textView.setStyleSheet("""
		font-family: SpoqaHanSans-Bold;
		font-size: 14px;
		font-weight: bold;
		font-style: normal;
		line-height: normal;
		color: #909090;
		"""
		)
		top3_textView.setAlignment(Qt.AlignLeft)
		top3_textView.setText("TOP 03.")
		top3_textView.move(self.title_width-238+self.offsetx1,578+self.offsety)
		top3_textView.adjustSize()
		
	
		self.setFixedSize(self.title_width, self.title_height+120)
		self.setWindowTitle('KAERI')
		self.start()
	
	def update_gui(self,ret,ind_to_class):	
		predict=ret['output']
		probs=predict.tolist()[0]
		top_n=(-predict).argsort()[0]
		top_n=top_n.tolist()
		sort_top_n={}
		for c in top_n:
			if ind_to_class[c] in sort_top_n:
				if(sort_top_n[ind_to_class[c]]<probs[c]):
					sort_top_n[ind_to_class[c]]=probs[c]
			else:
				sort_top_n[ind_to_class[c]]=probs[c]
				
		def f2(x):
			return x[1]
		sort_top_n=sorted(sort_top_n.items(),key=f2,reverse=True)
		sort_top_n=sort_top_n[:3]
		for i,item in enumerate(sort_top_n):
			key,val=item
			self.gui['top{}'.format(i+1)].setText("{}".format(key))
			self.gui['top{}'.format(i+1)].adjustSize()
			self.gui['top{}_val'.format(i+1)].setText("{:.2f}%".format(val*100))	
	
	def paintEvent(self,event):
		painter = QPainter()
		painter.begin(self)
		painter.setRenderHint(QPainter.Antialiasing)
		painter.setPen(QtGui.QColor(151,151,151))
		painter.setBrush(QtCore.Qt.white)
		painter.drawLine(246+self.offsetx, 540+self.offsety, 246+self.offsetx,605+self.offsety)
		painter.end()
		
	def start(self):
		self.th.daemon=True 
		self.th.start()
		print("started..")	
		
	def closeEvent(self, event):
		self.is_run=False
		self.th.join()
		event.accept()
		print("close_event")
		
	def keyPressEvent(self, e):
		interval=0.1
		if e.key() == Qt.Key_Up:
			zm=self.zoom_factor
			pre=zm+interval
			if(pre>1):
				self.zoom_factor=1
			else:
				self.zoom_factor+=interval
				
		if e.key() == Qt.Key_Down:
			zm=self.zoom_factor
			pre=zm-interval
			if(pre<0):
				self.zoom_factor=0
			else:
				self.zoom_factor-=interval
		factor=self.zoom_factor
		print("zoom_factor:{}".format(factor))

if __name__ == '__main__':
	opt = opts().init()  
	opt.default_resolution=[512,512] 
	app = QApplication(sys.argv)
	win = Demo(opt)
	win.show()
	sys.exit(app.exec_())