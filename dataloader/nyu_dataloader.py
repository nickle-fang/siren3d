#!/home/nickle/miniconda3/envs/siren3d/bin/python

from __future__ import print_function, division
import torch
import random
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF


local_pos = "/home/nickle/Desktop/siren3d/temp_files"
# local_pos = "/media/nickle/WD_BLUE"
server_pos = "../SSD"
use_pos = server_pos
if (use_pos == server_pos):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rescale(object):
    def __init__(self, output_size, data_gain):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.data_gain = data_gain

    def __call__(self, sample):
        rgb_img_raw, disp_gt = sample['rgb_img'], sample['disp_gt']

        if (self.data_gain):
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)
            scale = np.random.uniform(1.0, 1.5)
            
            if (flip > 0.5):
                rgb_img_raw = TF.hflip(rgb_img_raw)
                disp_gt = TF.hflip(disp_gt)
            rgb_img_raw = TF.rotate(rgb_img_raw, angle=degree, resample=Image.NEAREST)
            disp_gt = TF.rotate(disp_gt, angle=degree, resample=Image.NEAREST)
        
            rgb_img = transforms.CenterCrop(self.output_size)(rgb_img_raw)
            disp_gt = transforms.CenterCrop(self.output_size)(disp_gt)

        else:
            rgb_img = transforms.Resize(self.output_size)(rgb_img_raw)
            disp_gt = transforms.Resize(self.output_size)(disp_gt)


        return {'rgb_img': rgb_img, 
                'disp_gt': disp_gt}


class Rescale_Multiscale(object):
    def __init__(self, raw_size, size_scale2, size_scale4, size_scale8, data_gain=False):
        assert isinstance(size_scale2, tuple)
        self.size_scale2 = size_scale2
        self.size_scale4 = size_scale4
        self.size_scale8 = size_scale8
        self.data_gain = data_gain

    def __call__(self, sample):
        rgb_img_raw, disp_gt = sample['rgb_img'], sample['disp_gt']

        if (self.data_gain):
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)
            scale = np.random.uniform(1.0, 1.5)
            
            if (flip > 0.5):
                rgb_img_raw = TF.hflip(rgb_img_raw)
                disp_gt = TF.hflip(disp_gt)
            rgb_img_raw = TF.rotate(rgb_img_raw, angle=degree, resample=Image.NEAREST)
            disp_gt = TF.rotate(disp_gt, angle=degree, resample=Image.NEAREST)
        
            rgb_img = transforms.CenterCrop(self.size_scale2)(rgb_img_raw)
            disp_gt = transforms.CenterCrop(self.size_scale2)(disp_gt)

        else:
            rgb_scale2 = transforms.Resize(self.size_scale2)(rgb_img_raw)
            rgb_scale4 = transforms.Resize(self.size_scale4)(rgb_img_raw)
            rgb_scale8 = transforms.Resize(self.size_scale8)(rgb_img_raw)
            depth_scale2 = transforms.Resize(self.size_scale2)(disp_gt)
            depth_scale4 = transforms.Resize(self.size_scale4)(disp_gt)
            depth_scale8 = transforms.Resize(self.size_scale8)(disp_gt)

        return {'rgb_scale2': rgb_scale2, 
                'rgb_scale4': rgb_scale4, 
                'rgb_scale8': rgb_scale8, 
                'depth_scale2': depth_scale2,
                'depth_scale4': depth_scale4,
                'depth_scale8': depth_scale8}


class ToTensor(object):
    def __call__(self, sample):
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']

        return {'rgb_img': transforms.ToTensor()(rgb_img),
                'disp_gt': transforms.ToTensor()(disp_gt)}


class ToTensor_Multiscale(object):
    def __call__(self, sample):
        rgb_scale2 = sample['rgb_scale2']
        rgb_scale4 = sample['rgb_scale4']
        rgb_scale8 = sample['rgb_scale8']
        depth_scale2 = sample['depth_scale2']
        depth_scale4 = sample['depth_scale4']
        depth_scale8 = sample['depth_scale8']
        
        return {'rgb_scale2': transforms.ToTensor()(rgb_scale2), 
                'rgb_scale4': transforms.ToTensor()(rgb_scale4), 
                'rgb_scale8': transforms.ToTensor()(rgb_scale8), 
                'depth_scale2': transforms.ToTensor()(depth_scale2),
                'depth_scale4': transforms.ToTensor()(depth_scale4),
                'depth_scale8': transforms.ToTensor()(depth_scale8)}


class SamplePoint(object):
    def __init__(self, reso, sample_):
        self.resolution = reso
        self.sample_ = sample_

        self.index_list = np.array([[i, j] for i in range(reso[0]) for j in range(reso[1])]).transpose(1, 0)
        self.index_list = torch.from_numpy(self.index_list)

    def __call__(self, sample):
        rgb_img, disp_gt = sample['rgb_img'], sample['disp_gt']
        rgb_img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(rgb_img)

        disp_mask = disp_gt > 0

        index_list_masked = torch.masked_select(self.index_list.reshape\
                                                (2, self.resolution[0], self.resolution[1]), disp_mask)
        masked_points_num = int(index_list_masked.size()[0] / 2) # useful points number
        index_list_masked = index_list_masked.reshape(2, masked_points_num)
        random_choice = random.sample(range(masked_points_num), self.sample_)
        random_choice = sorted(random_choice)
        depth_sample_whole = torch.zeros_like(disp_gt)
        depth_sample_whole -= 1
        index_i = 0
        point_sample_index = torch.zeros(2, self.sample_)
        for p in random_choice:
            depth_sample_whole[:, index_list_masked[0, p].long(), index_list_masked[1, p].long()] = \
                disp_gt[:, index_list_masked[0, p].long(), index_list_masked[1, p].long()]
            point_sample_index[0, index_i] = index_list_masked[0, p]
            point_sample_index[1, index_i] = index_list_masked[1, p]
            index_i += 1

        return {'rgb_img':rgb_img, 
                'disp_gt':disp_gt, 
                'point_sample_index':point_sample_index, 
                'depth_sample_whole':depth_sample_whole, 
                'index_list': self.index_list}


class SamplePoint_Multiscale(object):
    def __init__(self, reso_scale8, sample_):
        self.resolution = reso_scale8
        self.sample_ = sample_

        self.index_list = np.array([[i, j] for i in range(reso_scale8[0]) for j in range(reso_scale8[1])]).transpose(1, 0)
        self.index_list = torch.from_numpy(self.index_list)

    def __call__(self, sample):
        rgb_scale2 = sample['rgb_scale2']
        rgb_scale4 = sample['rgb_scale4']
        rgb_scale8 = sample['rgb_scale8']
        depth_scale8 = sample['depth_scale8']
        depth_scale4 = sample['depth_scale4']
        depth_scale2 = sample['depth_scale2']

        rgb_scale2 = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(rgb_scale2)
        rgb_scale4 = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(rgb_scale4)
        rgb_scale8 = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(rgb_scale8)

        disp_mask = depth_scale8 > 0

        index_list_masked = torch.masked_select(self.index_list.reshape\
                                                (2, self.resolution[0], self.resolution[1]), disp_mask)
        masked_points_num = int(index_list_masked.size()[0] / 2) # useful points number
        index_list_masked = index_list_masked.reshape(2, masked_points_num)
        random_choice = random.sample(range(masked_points_num), self.sample_)
        random_choice = sorted(random_choice)
        depth_sample_whole = torch.zeros_like(depth_scale8)
        depth_sample_whole -= 1
        index_i = 0
        for p in random_choice:
            depth_sample_whole[:, index_list_masked[0, p].long(), index_list_masked[1, p].long()] = \
                depth_scale8[:, index_list_masked[0, p].long(), index_list_masked[1, p].long()]
            index_i += 1

        return {'rgb_scale2': rgb_scale2, 
                'rgb_scale4': rgb_scale4, 
                'rgb_scale8': rgb_scale8, 
                'depth_scale8': depth_scale8,
                'depth_scale4': depth_scale4,
                'depth_scale2': depth_scale2,
                'depth_sample_whole':depth_sample_whole}


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


class NYUDatasetAll(Dataset):
    def __init__(self, filelistpath, transform):
        self.transform = transform
        filelist = open(filelistpath, 'r')
        self.file_list = [row.replace("\n", "") for row in filelist]
        filelist.close()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        f = h5py.File(use_pos + self.file_list[index], 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        sample = {'rgb_img':rgb, 'disp_gt':dep}

        if self.transform:
            sample = self.transform(sample)

        return sample


# if __name__ == "__main__":
#     rootdir = "/home/nickle/Desktop/siren3d/temp_files/nyudepthv2"
#     first_dir_list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
#     val_dir = rootdir + '/' + first_dir_list[0] + '/official'
#     train_dir = rootdir + '/' + first_dir_list[1]
#     train_dir_list = os.listdir(train_dir)


#     print(train_dir)
#     print(val_dir)

#     for root, dirs, files in os.walk(val_dir):
#         for name in files:
#             print(name)
        # for name in dirs:
        #     print(name)



    # dirs = os.walk(rootdir)
    # for dir in dirs:
    #     print(dir)
    # print(files)


#     nyu_dataset = NYUDataset(transform=transforms.Compose([Rescale((224,224)), ToTensor()]))

#     for i in range(2):
#         sample = nyu_dataset[i]
#         rgb_img = sample['rgb_img']base_bugfixed_baseline.tar
#         disp_gt = sample['disp_gt']

#         print(disp_gt)
