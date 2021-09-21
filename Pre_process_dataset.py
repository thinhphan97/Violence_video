import numpy as np
import glob
import pandas as pd
import os
from sklearn.utils import shuffle

def data_split(data_dir = "dataset/extract/", slice_sample = 10, stride=3):
	#data_dir = "dataset/extract/"
	#stride = 2
	#slice_sample = 10
	result_dict = {}
	result_dict["class"] = []
	result_dict["images"] = []
	result_dict["video_name"] = []
	name_class = ["NonViolence", "Violence"]
	for Class in name_class:
		name_folder = glob.glob(os.path.join(data_dir,Class)+"/*")
		for folder in name_folder:
#			print(folder)
			images = glob.glob(folder + "/*" )
			images = sorted(images)
			length = len(images)
			image_result = []
			for i in range(length-(slice_sample*stride -2)):
				image_slice = []
				for j in range(0,slice_sample*stride,stride):
					image_slice.append(images[i+j])
				result_dict["images"].append(image_slice)
				result_dict["class"].append(Class)
				result_dict["video_name"].append(folder.split("/")[3])
				#print(len(image_slice))
			#print(len(image_result))
#			break
	return pd.DataFrame.from_dict(result_dict)
def train_test_val_split(df, train_percent = 0.7, val_percent = 0.15 ):

	NV_df = shuffle(df[df["class"]=="NonViolence"]).reset_index(drop=True)
	V_df = shuffle(df[df["class"]=="Violence"]).reset_index(drop=True)

	train_NV_df = NV_df.sample(frac=train_percent, random_state = 200)
	train_V_df = V_df.sample(frac=train_percent, random_state = 200)
	
	test_NV_df = NV_df.drop(train_NV_df.index).reset_index().sample(frac=0.5, random_state = 200)
	test_V_df = V_df.drop(train_V_df.index).reset_index().sample(frac=0.5, random_state = 200)

	
	val_NV_df =  NV_df.drop(train_NV_df.index).reset_index().drop(test_NV_df.index)
	val_V_df = V_df.drop(train_V_df.index).reset_index().drop(test_V_df.index)

	train_df = shuffle(train_NV_df.append(train_V_df)).reset_index(drop = True)
	test_df = shuffle(test_NV_df.append(test_V_df)).reset_index(drop = True)
	val_df = shuffle(val_NV_df.append(val_V_df)).reset_index(drop = True)	

	print(len(df))
	print(len(train_df))
	print(len(test_df))
	print(len(val_df))
	train_df.to_pickle("dataset/train.pkl")
	test_df.to_pickle("dataset/test.pkl")
	val_df.to_pickle("dataset/val.pkl")
if __name__ == "__main__":
	train_test_val_split(data_split())
	print(pd.read_pickle("dataset/train.pkl"))
