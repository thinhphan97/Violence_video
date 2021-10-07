import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
import numpy as np
from albumentations import pytorch
import albumentations as A
import cv2
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
        self.totensor = pytorch.transforms.ToTensor(
                        normalize={"mean": self.cfg.DATA.MEAN,
                                    "std": self.cfg.DATA.STD})
        if self.mode == "train":
            self.train_aug = A.Compose([
                A.RandomResizedCrop(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE,
                                    interpolation=cv2.INTER_LINEAR, scale=(0.8, 1)),
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
                ]),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(p=1.),
                    A.GaussNoise(p=1.),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.MedianBlur(blur_limit=3, p=1.),
                    A.Blur(blur_limit=3, p=1.),
                    A.NoOp()
                ])
            ])

    def _load_img(self, file_path):
        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            img = self.train_aug(image=img)["image"]

        # Normalize by ImageNet statistics
        img_tensor = self.totensor(image=img)["image"]
        return img_tensor

