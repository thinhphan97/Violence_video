import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
import numpy as np
from albumentations import pytorch
import albumentations as A
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Dataset_Custom(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.CLASSES = self.cfg.CONST.LABELS
        self.mode = mode
        self.train_df = pd.read_pickle(self.cfg.DIRS.TRAIN_DF)
        self.valid_df = pd.read_pickle(self.cfg.DIRS.VALID_DF)
        self.test_df = pd.read_pickle(self.cfg.DIRS.TEST_DF)
        self.totensor = pytorch.transforms.ToTensorV2()
                        # normalize={"mean": self.cfg.DATA.MEAN,
                        #             "std": self.cfg.DATA.STD})
        if self.mode == "train":
            self.train_aug = A.Compose([
                # A.RandomResizedCrop(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE,
                #                     interpolation=cv2.INTER_LINEAR, scale=(0.8, 1)),
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
                    A.NoOp()
                ])
                # A.OneOf([
                #     A.IAAAdditiveGaussianNoise(p=1.),
                #     A.GaussNoise(p=1.),
                #     A.NoOp()
                # ]),
                # A.OneOf([
                #     A.MedianBlur(blur_limit=3, p=1.),
                #     A.Blur(blur_limit=3, p=1.),
                #     A.NoOp()
                # ])
            ])
        self.label_one_hot = self._label_2_one_hot()

    def _label_2_one_hot(self):
        labels = self.CLASSES
        labels = np.array(labels)
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        dict_labels ={}
        for (i,label) in enumerate(labels):
            dict_labels[label]= onehot_encoded[i]
        return dict_labels

    def _load_img(self, file_path):
        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.DATA.IMG_SIZE,self.cfg.DATA.IMG_SIZE), interpolation = cv2.INTER_AREA)
        if self.mode == "train":
            img = self.train_aug(image=img)["image"]

        # Normalize by ImageNet statistics
        img_tensor = self.totensor(image=img)["image"]
        return img_tensor/255

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
        labels = data["class"]
        labels = self.label_one_hot[labels]
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