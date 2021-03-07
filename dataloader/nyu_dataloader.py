#!/usr/bin/python3

from __future__ import print_function, division
import torch
import random
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms

from model import fc_basefunction as fc


local_pos = "/media/nickle/WD_BLUE/NYUDataV2"
server_pos = "../SSD"
use_pos = local_pos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']
        
        new_h, new_w = self.output_size

        rgb_img = transform.resize(rgb_img, (new_h, new_w))
        disp_gt = transform.resize(disp_gt, (new_h, new_w))

        return {'rgb_img': rgb_img, 'disp_gt': disp_gt}


class ToTensor(object):
    def __call__(self, sample):
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']

        rgb_img = rgb_img.transpose((2, 0, 1))
        return {'rgb_img': torch.from_numpy(rgb_img),
                'disp_gt': torch.from_numpy(disp_gt).unsqueeze(0)}


class SamplePoint(object):
    def __init__(self, reso, sample_):
        self.resolution = reso
        self.sample_ = sample_

    def __call__(self, sample):
        point_sample_index = random.sample([[i, j] for i in range(self.resolution[0]) for j in range(self.resolution[1])], self.sample_)
        point_sample_index = sorted(point_sample_index)
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']

        depth_sample_whole = torch.zeros_like(disp_gt)
        for p in point_sample_index:
            depth_sample_whole[:, p[0], p[1]] = disp_gt[:, p[0], p[1]]

        return {'rgb_img':rgb_img, 'disp_gt':disp_gt, 
                'point_sample_index':point_sample_index, 'depth_sample_whole':depth_sample_whole}


class NYUDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return 1448
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        rgb_img = io.imread(use_pos + "/rgb/" + str(index) + ".jpg")
        disp_gt = io.imread(use_pos + "/depth/" + str(index) + ".png")

        sample = {'rgb_img':rgb_img, 'disp_gt':disp_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample


# if __name__ == "__main__":
#     nyu_dataset = NYUDataset(transform=transforms.Compose([Rescale((224,224)), ToTensor()]))

#     for i in range(2):
#         sample = nyu_dataset[i]
#         rgb_img = sample['rgb_img']
#         disp_gt = sample['disp_gt']

#         print(disp_gt)
