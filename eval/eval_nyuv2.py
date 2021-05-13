#!/home/nickle/miniconda3/envs/siren3d/bin/python

import torch
import time
import os
import cv2
import math
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
shuffle_bool = False
eval_bool = True
eval_pics = 1448
sample_num = 1000
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (240, 320)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss_function = torch.nn.MSELoss()

def position_encoding(tensor):
    x, y = torch.split(tensor, 1, dim=1)
    encoded = torch.cat((x, x, y, y), dim=1)
    for i in range(4):
        encoded = torch.cat((encoded, 
                            torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
                            torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

    return encoded.permute(0, 2, 1)


def least_squres(tensor_a, tensor_b):
    fc_concat_trans = tensor_a.permute(1, 0) # final_layers * sample_num
    new_a = torch.matmul(fc_concat_trans, tensor_a)
    new_b = torch.matmul(fc_concat_trans, tensor_b)
    w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(final_layer+20).to(device)), new_b) # final_layers * 1

    return w


def eval_main_loop(dataloader, feature_ex_net, base_func):
    it = iter(dataloader)
    total_rmse = 0.0
    time_start = time.time()

    for i in range(eval_pics):
        sample_dict = next(it)

        # prepare the data
        rgb_img = sample_dict['rgb_img']
        disp_gt = sample_dict['disp_gt']
        point_sample_index = sample_dict['point_sample_index']
        depth_sample_whole = sample_dict['depth_sample_whole'] # not-sampled points are zero, size as disp_gt
        index_list = sample_dict['index_list']

        rgb_img = rgb_img.to(device, torch.float32)
        rgb_img_scale2 = rgb_img_scale2.to(device, torch.float32)
        rgb_img_scale4 = rgb_img_scale4.to(device, torch.float32)
        disp_gt = disp_gt.to(device, torch.float32)
        depth_sample_whole = depth_sample_whole.to(device, torch.float32)
        index_list = index_list.squeeze(0).to(device, torch.float32)

        sample_mask = depth_sample_whole > 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

        feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution

        feature_map_sample = torch.masked_select(feature_map, sample_mask) # 256000
        feature_map_sample = feature_map_sample.reshape(1, feature_map_layers, sample_num).permute(0, 2, 1) # 1*1000*256

        # point_sample_index_norm ： 1*2*sample_num
        point_sample_index_norm = torch.from_numpy(np.array(point_sample_index, dtype=np.float32).transpose(1, 0))
        point_sample_index_norm = point_sample_index_norm.to(device, dtype=torch.float32) / \
                                    torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
        point_sample_index_norm = point_sample_index_norm.unsqueeze(0).to(device, dtype=torch.float32)

        fc_in = torch.cat((feature_map_sample, position_encoding(point_sample_index_norm)), dim=2) # 1*1000*(256+40)
        fc_in = fc_in.to(device).squeeze()

        fc_concat = base_func(fc_in) # 1 * sample_num * final_layers

        # add pose again
        fc_concat = torch.cat((fc_concat, position_encoding(point_sample_index_norm).squeeze()), dim=1)

        # caculate w
        w = least_squres(fc_concat, depth_sample)

        # predict depth using W
        feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)

        fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()

        fc_pic_out = base_func(fc_pic_in)

        # add pose again
        fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)

        predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

        rmse_single = loss_function(predict_depth, disp_gt)
        total_rmse += rmse_single
        print("Single picture loss is :", rmse_single)
    print("RMSE is: ", total_rmse / eval_pics)
    print("Using time: ", time.time() - time_start)


def eval_main():
    base_func = fc.FC_basefunction(final_layer, feature_map_layers+20)
    base_func.load_state_dict(torch.load("./checkpoints/base_func_baseline.tar"))
    base_func = base_func.to(device)

    feature_ex_net = fc.DRPNet()
    feature_ex_net.load_state_dict(torch.load("./checkpoints/unet_baseline.tar"))
    feature_ex_net = feature_ex_net.to(device)

    nyu_eval_dataset = nyu.NYUDataset(filelistpath="./data/NYUv2_eval.txt",
                                      transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                                size_scale2=(100, 100),
                                                                                size_scale4=(100, 100)), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=resolution, 
                                                                                    sample_=sample_num)]))
    eval_dataloader = DataLoader(nyu_eval_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    base_func.eval()
    feature_ex_net.eval()
    eval_main_loop(eval_dataloader, feature_ex_net, base_func)
        
if __name__ == '__main__':
    eval_main()
