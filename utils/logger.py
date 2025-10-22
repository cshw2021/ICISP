import os
import logging
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

import glob
import random

import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset

class ImageFolder(Dataset):

    def __init__(self, root, dataset_name, patch_size, split="train"):
        if split == 'train':
            self.samples = sorted(glob.glob(os.path.join(root, dataset_name, split, '*.png')))
        elif split == 'test':
            self.samples = sorted(glob.glob(os.path.join(root, dataset_name, '*.png')))
        print('{} stage: total samples:{}'.format(split, len(self.samples)))

        self.patch_size = patch_size
        self.split = split

    def __getitem__(self, index):
        img = cv2.imread(self.samples[index]).astype(np.float32) / 255.

        H, W, _ = img.shape
        if self.split == 'train':
            randn_h = random.randint(0, H - self.patch_size)
            randn_w = random.randint(0, W - self.patch_size)
            img = img[randn_h:randn_h + self.patch_size, randn_w:randn_w + self.patch_size, :]

        if self.split == 'test':
            if self.patch_size is not None:
                begin_h = (H - self.patch_size) // 2
                begin_w = (W - self.patch_size) // 2
                img = img[begin_h:begin_h + self.patch_size, begin_w:begin_w + self.patch_size, :]

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1)))).float()

        return img

    def __len__(self):
        return len(self.samples)

