from typing import final
from numpy.lib.function_base import disp
import torch
import random
import time
import cv2
import math
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
shuffle_bool = True
total_pics = 1448
total_epochs = 50000
sample_num = 1000
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (240, 320)
lamda1 = 5
lamda2 = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def position_encoding(tensor):
    x, y = torch.split(tensor, 1, dim=1)
    encoded = torch.cat((torch.sin((2**0)*math.pi*x), torch.cos((2**0)*math.pi*x),
                        torch.sin((2**0)*math.pi*y), torch.cos((2**0)*math.pi*y)), dim=1)
    for i in range(9):
        encoded = torch.cat((encoded, 
                            torch.sin((2**(i+1))*math.pi*x), torch.cos((2**(i+1))*math.pi*x),
                            torch.sin((2**(i+1))*math.pi*y), torch.cos((2**(i+1))*math.pi*y)), dim=1)

    return encoded


def train_main():
    base_func = fc.FC_basefunction(final_layer, feature_map_layers+40)
    # base_func.load_state_dict(torch.load("./checkpoints/256_w_0.tar"))
    base_func = base_func.to(device)
    base_func.train()

    feature_ex_net = fc.DRPNet()
    # feature_ex_net.load_state_dict(torch.load("./checkpoints/256_w_0.tar"))
    feature_ex_net = feature_ex_net.to(device)
    feature_ex_net.train()

    base_func_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.00001, weight_decay=5e-4)
    feature_ex_net_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.00001, weight_decay=5e-4)

    writer1 = SummaryWriter(log_dir='runs/test_normal_structure')

    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.L1Loss()

    for epoch in range(total_epochs):
        # flying_dataset = fly.FlyingDataset(filelistpath="./data/flying3d_train.txt",
        #                                     transform=transforms.Compose([
        #                                         fly.Rescale((resolution, resolution))
        #                                     ]))
        # dataloader = DataLoader(flying_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
        # it = iter(dataloader)

        nyu_dataset = nyu.NYUDataset(transform=transforms.Compose([ nyu.Rescale((resolution[0],resolution[1])), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=resolution, 
                                                                                    sample_=sample_num)]))
        dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)
        it = iter(dataloader)

        for i in range(total_pics):
            time_start = time.time()

            # calculate the W using known depth
            sample_dict = next(it)
            rgb_img = sample_dict['rgb_img']
            disp_gt = sample_dict['disp_gt']
            point_sample_index = sample_dict['point_sample_index']
            depth_sample_whole = sample_dict['depth_sample_whole'] # not-sampled points are zero, size as disp_gt

            rgb_img = rgb_img.to(device, torch.float32)
            disp_gt = disp_gt.to(device, torch.float32)
            depth_sample_whole = depth_sample_whole.to(device, torch.float32)

            sample_mask = depth_sample_whole > 0
            depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

            time_1 = time.time()

            feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution

            feature_map_sample = torch.masked_select(feature_map, sample_mask) # 256000
            feature_map_sample = feature_map_sample.reshape(1, sample_num, feature_map_layers) # 1*1000*256

            time_2 = time.time()

            # point_sample_index_norm ： 1*2*sample_num
            point_sample_index_norm = torch.from_numpy(np.array(point_sample_index).transpose(1, 0))
            point_sample_index_norm = point_sample_index_norm.to(device, dtype=torch.float32) / \
                                        torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
            point_sample_index_norm = point_sample_index_norm.unsqueeze(0).to(device, dtype=torch.float32)

            fc_in = torch.cat((feature_map_sample, position_encoding(point_sample_index_norm).permute(0, 2, 1)), dim=2) # 1*(256+40)*1000
            fc_in = fc_in.to(device).squeeze()

            time_3 = time.time()

            fc_concat = base_func(fc_in) # 1 * sample_num * final_layers

            fc_concat_trans = fc_concat.permute(1, 0) # final_layers * sample_num
            new_a = torch.matmul(fc_concat_trans, fc_concat)
            new_b = torch.matmul(fc_concat_trans, depth_sample)
            w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(final_layer).to(device)), new_b) # final_layers * 1

            predict_sample_depth = torch.matmul(fc_concat, w)

            time_4 = time.time()

            loss_1 = loss_function(predict_sample_depth, depth_sample)

            time_5 = time.time()

            # predict depth using W
            feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)
            index_list = np.array([[i / resolution[0], j / resolution[1]] \
                                for i in range(resolution[0]) for j in range(resolution[1])]).transpose(1, 0)
            index_list = torch.from_numpy(index_list).unsqueeze(0).to(device, torch.float32)
            
            fc_pic_in = torch.cat((feature_map, position_encoding(index_list).permute(0, 2, 1)), axis=2).squeeze()

            fc_pic_out = base_func(fc_pic_in)
            
            predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

            loss_2 = loss_function(predict_depth, disp_gt)

            time_6 = time.time()
            #########################################################################################
            loss = lamda1 * loss_1 + lamda2 * loss_2

            base_func_optimizer.zero_grad()
            feature_ex_net_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_func.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(feature_ex_net.parameters(), 10)
            base_func_optimizer.step()
            feature_ex_net_optimizer.step()

            writer1.add_scalar('loss', loss.item(), global_step=i+total_pics*epoch)
            writer1.add_scalar('loss_1', loss_1.item(), global_step=i+total_pics*epoch)
            writer1.add_scalar('loss_2', loss_2.item(), global_step=i+total_pics*epoch)

            disp_gt = disp_gt.squeeze().cpu().detach().numpy()
            predict_depth = predict_depth.squeeze().cpu().detach().numpy()

            # disp_color = cv2.applyColorMap(disp_gt.astype(np.uint8), 4)
            # predict_color = cv2.applyColorMap(predict_depth.astype(np.uint8), 4)

            # disp_gt = cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # predict_depth = cv2.normalize(predict_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            if (i % 10 == 0):
                writer1.add_image('groundtruth and out', 
                                    np.concatenate((disp_gt, predict_depth), axis=1, out=None),
                                    global_step=i+total_pics*epoch,
                                    dataformats='HW')

            print('Epoch:'+str(epoch)+' Pic:'+str(i)+" Processing a single picture using: %.2f s"%(time.time()-time_start))
            print("---------loss             : %.4f" % loss.item())
            print("---------loss_1           : %.4f" % loss_1.item())
            print("---------loss_2           : %.4f" % loss_2.item())

            # print("0-1: %.4f " % (time_1 - time_start))
            # print("1-2: %.4f " % (time_2 - time_1))
            # print("2-3: %.4f " % (time_3 - time_2)) #######
            # print("3-4: %.4f " % (time_4 - time_3))
            # print("4-5: %.4f " % (time_5 - time_4))
            # print("5-6: %.4f " % (time_6 - time_5)) ########
            # print("6-7: %.4f " % (time.time() - time_6)) ########
        
        if (epoch % 10 == 0):
            savefilename = './checkpoints/base_func_'+str(epoch)+'.tar'
            torch.save(base_func.state_dict(), savefilename)
            savefilename = './checkpoints/unet_'+str(epoch)+'.tar'
            torch.save(feature_ex_net.state_dict(), savefilename)
            

if __name__ == '__main__':
    train_main()
