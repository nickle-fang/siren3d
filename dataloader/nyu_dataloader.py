#!/home/nickle/miniconda3/envs/siren3d/bin/python

from __future__ import print_function, division
import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from model import fc_basefunction as fc


local_pos = "/media/nickle/WD_BLUE/NYUDataV2"
server_pos = "../SSD/NYUDataV2"
use_pos = server_pos
if (use_pos == server_pos):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rescale(object):
    def __init__(self, output_size, size_scale2, size_scale4):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.size_scale2 = size_scale2 
        self.size_scale4 = size_scale4

    def __call__(self, sample):
        rgb_img_raw, disp_gt = sample['rgb_img'], sample['disp_gt']

        rgb_img = transforms.Resize(self.output_size)(rgb_img_raw)
        disp_gt = transforms.Resize(self.output_size)(disp_gt)

        return {'rgb_img': rgb_img, 
                'disp_gt': disp_gt}


class ToTensor(object):
    def __call__(self, sample):
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']

        # return {'rgb_img': torch.from_numpy(rgb_img).permute(0, 2, 1),
        #         'disp_gt': torch.from_numpy(disp_gt).unsqueeze(0)}

        return {'rgb_img': transforms.ToTensor()(rgb_img),
                'disp_gt': transforms.ToTensor()(disp_gt)}


class SamplePoint(object):
    def __init__(self, reso, sample_):
        self.resolution = reso
        self.sample_ = sample_

        self.index_list = np.array([[i / reso[0], j / reso[1]] \
                            for i in range(reso[0]) for j in range(reso[1])]).transpose(1, 0)
        self.index_list = torch.from_numpy(self.index_list).unsqueeze(0)

    def __call__(self, sample):
        point_sample_index = random.sample([[i, j] for i in range(self.resolution[0]) for j in range(self.resolution[1])], self.sample_)
        point_sample_index = sorted(point_sample_index)
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']

        depth_sample_whole = torch.zeros_like(disp_gt)
        for p in point_sample_index:
            depth_sample_whole[:, p[0], p[1]] = disp_gt[:, p[0], p[1]]

        return {'rgb_img':rgb_img, 
                'disp_gt':disp_gt, 
                'point_sample_index':point_sample_index, 
                'depth_sample_whole':depth_sample_whole,
                'index_list': self.index_list}


class NYUDataset(Dataset):
    def __init__(self, filelistpath, transform):
        self.transform = transform
        filelist = open(filelistpath, "r")
        self.file_list = [row.replace("\n", "") for row in filelist]
        filelist.close()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        rgb_img = Image.open(use_pos + "/rgb/" + str(index) + ".jpg")
        disp_gt = Image.open(use_pos + "/depth/" + str(index) + ".png")

        sample = {'rgb_img':rgb_img, 'disp_gt':disp_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample


# if __name__ == "__main__":
#     nyu_dataset = NYUDataset(transform=transforms.Compose([Rescale((224,224)), ToTensor()]))

#     for i in range(2):
#         sample = nyu_dataset[i]
#         rgb_img = sample['rgb_img']base_bugfixed_baseline.tar
#         disp_gt = sample['disp_gt']

#         print(disp_gt)
