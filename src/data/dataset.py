import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
import numpy as np
from albumentations import pytorch
import albumentations as A
from torchvision import transforms
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Dataset_Custom(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.CLASSES = self.cfg.CONST.LABELS
        self.mode = mode
        self.train_df = self._data_split(self.cfg.DIRS.TRAIN_DF,self.cfg.DATA.NUM_SLICES,cfg.DATA.STRIDE)
        self.valid_df = self._data_split(self.cfg.DIRS.VALID_DF,self.cfg.DATA.NUM_SLICES,cfg.DATA.STRIDE)
        self.test_df =  self._data_split(self.cfg.DIRS.TEST_DF,self.cfg.DATA.NUM_SLICES,cfg.DATA.STRIDE)
        self.totensor = pytorch.transforms.ToTensorV2()
        self.normalize = A.Compose([A.Normalize()])
        if self.mode == "train":
            self.train_aug = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1.),
                    A.VerticalFlip(p=1.),
                ]),
                A.OneOf([
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=30,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.GridDistortion(
                        distort_limit=0.2,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.OpticalDistortion(
                        distort_limit=0.2,
                        shift_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.NoOp()]),
                A.OneOf([
                    A.MedianBlur(blur_limit=3, p=1.),
                    A.Blur(blur_limit=3, p=1.),
                    A.NoOp()
                ])
            ])
    def _data_split(self, data_dir, slice_sample , stride):
        
        result_dict = {}
        result_dict["class"] = []
        result_dict["images"] = []
        result_dict["video_name"] = []
        result_dict["labels"] = []
        name_class = ["abnormal", "normal"]
        folders = glob.glob(data_dir+"*")
        for Class in name_class:
            name_folder = glob.glob(os.path.join(data_dir,Class)+"/*")
            for folder in name_folder:
                images = glob.glob(folder + "/*" )
                images = sorted(images)
                length = len(images)
                for i in range(0, length-(slice_sample*stride -2),self.cfg.DATA.STEP):
                    image_slice = []
                    for j in range(0,slice_sample*stride,stride):
                        image_slice.append(images[i+j])
                    result_dict["images"].append(image_slice)
                    result_dict["class"].append(Class)
                    if Class == "abnormal":
                        label = [1., 0.]
                    else:
                        label = [0., 1.]
                    result_dict["labels"].append(label)
                    result_dict["video_name"].append(folder.split("/")[2])
        return pd.DataFrame.from_dict(result_dict)

    def _load_img(self, file_path):
        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.DATA.IMG_SIZE,self.cfg.DATA.IMG_SIZE), interpolation = cv2.INTER_AREA)
        if self.mode == "train":
            img = self.train_aug(image=img)["image"]
        img  = self.normalize(image=img)["image"]
        img_tensor = self.totensor(image=img)["image"]
        return img_tensor

class Dataset_Custom_3d(Dataset_Custom):

    def __init__(self,cfg, mode='train'):
        super(Dataset_Custom_3d, self).__init__(cfg, mode)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_df)
        elif self.mode == "valid":
            return len(self.valid_df)
        elif self.mode == "test":
            return len(self.test_df)
    
    def _load_study(self,data_df, idx):
        data = data_df.iloc[idx]
        img_names = data["images"]
        imgs = [self._load_img(img_path) for img_path in img_names]
        imgs = torch.stack(imgs)
        labels = np.array(data["labels"])
        if self.mode == "train" or self.mode == "valid":
            return imgs, torch.from_numpy(labels).type('torch.FloatTensor')

        elif self.mode == "test":
            return imgs, labels

    def __getitem__(self, idx):
        if self.mode == "train":
            return self._load_study(self.train_df,idx)
        elif self.mode == "valid":
            return self._load_study(self.valid_df,idx)
        elif self.mode == "test":
            return self._load_study(self.test_df,idx)

class Dataset_Custom_2d(Dataset_Custom):

    def __init__(self,cfg, mode='train'):
        super(Dataset_Custom_2d, self).__init(cfg, mode)
    
    def __getitem__(self, idx):
        pass