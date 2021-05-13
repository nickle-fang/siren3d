#!/usr/bin/python3

import torch
import re
import cv2
import numpy as np
import random
from torch.utils.data import Dataset


local_pos = "/media/nickle/WD_BLUE/stereo_data"
server_pos = "../SSD"
use_pos = local_pos

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #
    img = np.reshape(dispariy, newshape=(height, width, channels))
    return img


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        left_img, disp_gt = sample['left_img'], sample['disp_gt']

        if isinstance(self.output_size, tuple):
            new_h, new_w = self.output_size
            h_bias = random.randint(0, 316)
            w_bias = random.randint(0, 736)
            left_img = left_img[:, h_bias:(new_h+h_bias), w_bias:(new_w+w_bias)]
            disp_gt = disp_gt[:, h_bias:(new_h+h_bias), w_bias:(new_w+w_bias)]
        else:
            print("Error!!! Please input a tuple!")
        
        return {'left_img':left_img, 'disp_gt':disp_gt}


class FlyingDataset(Dataset):
    def __init__(self, filelistpath, transform=None):
        self.transform = transform
        filelist = open(filelistpath, "r")
        self.file_list = [row.replace("\n", "") for row in filelist]
        filelist.close()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        left_img = cv2.imread(use_pos + self.file_list[index]%("frames_finalpass", "left") + ".png")
        disp_gt = read_pfm(use_pos + self.file_list[index]%("disparity", "left") + ".pfm")

        # left_img = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        # disp_gt = Image.fromarray(cv2.cvtColor(disp_gt, cv2.COLOR_BGR2RGB))

        left_img = left_img.transpose((2, 0, 1))
        disp_gt = disp_gt.transpose((2, 0, 1))

        left_img = torch.from_numpy(left_img).float()
        disp_gt = torch.from_numpy(np.flip(disp_gt, 1).copy()).float()

        sample = {'left_img':left_img, 'disp_gt':disp_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
