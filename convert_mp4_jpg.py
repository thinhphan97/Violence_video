


import cv2
import numpy as np
import glob
import os
from src.config import get_cfg
def read_mp4(link_save, link_file,cfg):
	vidcap = cv2.VideoCapture(link_file)
	success,image = vidcap.read()

	count = 0
	while success:
		image = cv2.resize(image, (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE), interpolation = cv2.INTER_AREA)
		cv2.imwrite(link_save+"/frame_{}.jpg".format(str(count).zfill(4)), image)
		success,image = vidcap.read()
		count +=1
	print('[save] ' + link_save + ".")

def preprocess():
	cfg = get_cfg()
	os.makedirs("dataset/extract1",exist_ok=True)
	list_folder = glob.glob("dataset/data_stream/*")
	for link in list_folder:
		list_file  = glob.glob(link+"/*")
		link_folder = link.split('/')[0]+"/extract1/"+link.split('/')[2]
		os.makedirs(link_folder,exist_ok=True)
		for  file  in  list_file:
			link_save = link_folder + "/" + file.split('/')[3].split('.')[0]
			os.makedirs(link_save,exist_ok=True)
			read_mp4(link_save,file,cfg)

if __name__ == "__main__":
	preprocess()
