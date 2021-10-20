
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from src.data import Dataset_Custom_3d

if __name__ == "__main__":
	# import pandas as pd
	# import matplotlib.pyplot as plt
	# import cv2

	'''df = pd.read_pickle("dataset/val.pkl")
	# df = df["class"]
	df["value"] = 1
	df = df.groupby(["class"]).sum()
	#print(df)
	
	
	# plot = df.plot.pie(y='mass', figsize=(5, 5))
	# plt.figure(figsize=(8,8))
	# ax1 = plt.subplot(121, aspect='equal')
	df.plot(kind='pie', y='value',autopct='%1.1f%%', 
	startangle=90, shadow=False, labels=['abnormal','normal'],
 	legend = False, fontsize=20)
	plt.show()
	print(df.loc[0])'''
	# string ="dataset\extract\normal\video_1\frame_0243.jp".replace("\", "/")
	# print(string)
	# img = cv2.imread("dataset/extract/normal/video_1/frame_0498.jpg")
	# print(img.shape)
	# from src.config import get_cfg
	# from src.modeling import ConvLSTM3D
	# import torch
	# cfg = get_cfg()

	# print(cfg)
	
	# x = torch.rand((32, 10, 3, 128, 128))
	# bsize, seq_len, c, h, w = x.size()
	# x = x.view(bsize * seq_len, c, h, w)
    
    # # convlstm = ConvLSTM(cfg)
	# convlstm3d = ConvLSTM3D(cfg)
	# last_states = convlstm3d(x, seq_len)
	# h = last_states
	# print(h.size())
	# print(h)
	from src.data import Dataset_Custom_3d
	from yacs.config import CfgNode as CN
	from torch.utils.data import DataLoader
	from tqdm import tqdm

	cfg = CN()
	cfg.DIRS = CN()
	cfg.DIRS.TRAIN_DF = "dataset/train.pkl"
	cfg.DIRS.VALID_DF = "dataset/val.pkl"
	cfg.DIRS.TEST_DF = "dataset/test.pkl"
	cfg.DATA = CN()
	cfg.DATA.IMG_SIZE = 128
	cfg.CONST = CN()
	cfg.CONST.LABELS = [
	"normal", "abnormal"
	]
	cfg.TRAIN = CN()
	cfg.TRAIN.BATCH_SIZE = 30
	Dataset = Dataset_Custom_3d
	data = Dataset(cfg,mode='train')
	train_loader = DataLoader(data, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=True,
                            drop_last=False, num_workers= 3)
	tbar = tqdm(train_loader)
	for i, (image, target) in enumerate(train_loader):
		tbar.set_description(f"image shape: {image.shape}, target shape: {target.shape}")

	# image, target = train_loader
	# print(image.shape)
	# print(target.shape)