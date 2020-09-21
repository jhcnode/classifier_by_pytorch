import os.path as osp
import sys
import os
import logging

		
def add_path(path):
	if 'D:\\tensorflow\\models\\research' in sys.path:
		sys.path.remove('D:\\tensorflow\\models\\research')
	if 'D:\\tensorflow\\models\\research\\slim' in sys.path:
		sys.path.remove('D:\\tensorflow\\models\\research\\slim')
	if 'H:\\KAERI\\Custom' in sys.path:
		sys.path.remove('H:\\KAERI\\Custom')	
	if path not in sys.path:
		sys.path.insert(0, path)

		

this_dir = osp.dirname(__file__)

# # Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')

add_path(lib_path)

# print(sys.path)

