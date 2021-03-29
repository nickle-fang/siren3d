#!/home/nickle/miniconda3/envs/siren3d/bin/python

import torch
import time
import os
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
shuffle_bool = False
eval_bool = True
train_pics = 1200
eval_pics = 248
total_epochs = 50000
sample_num = 1000
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (240, 320)
reso_scale2 = (120, 160)
reso_scale4 = (64, 88)
lamda1 = 1 # loss_1
lamda2 = 1 # loss_2
berHu_thredshold = 0.1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_file_name = "test"

# define loss function
# loss_function = torch.nn.SmoothL1Loss(beta=0.001)
loss_function = torch.nn.L1Loss()

upsample_visual = torch.nn.Upsample(size=(resolution[0], resolution[1]), mode="bilinear", align_corners=True)

def position_encoding(tensor):
    x, y = torch.split(tensor, 1, dim=1)
    # encoded = torch.cat((torch.sin((2**0)*math.pi*x), torch.cos((2**0)*math.pi*x),
    #                     torch.sin((2**0)*math.pi*y), torch.cos((2**0)*math.pi*y)), dim=1)
    encoded = torch.cat((x, x, y, y), dim=1)
    for i in range(4):
        # encoded = torch.cat((encoded, 
        #                     torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
        #                     torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

        encoded = torch.cat((encoded, x, x, y, y), dim=1)

    return encoded.permute(0, 2, 1)


def berHu_loss(tensor1, tensor2, c):
    diff = torch.abs(tensor1 - tensor2)
    berhu = (diff**2 + c**2) / (2*c)
    mask = diff <= c
    mask_ = diff > c
    loss = diff * mask + berhu * mask_

    return loss.sum() / tensor1.numel()


def least_squres(tensor_a, tensor_b):
    fc_concat_trans = tensor_a.permute(1, 0) # final_layers * sample_num
    new_a = torch.matmul(fc_concat_trans, tensor_a)
    new_b = torch.matmul(fc_concat_trans, tensor_b)
    w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(final_layer+20).to(device)), new_b) # final_layers * 1

    return w


def train_main_loop(dataloader, feature_ex_net, base_func, feature_ex_net_optimizer, \
                    base_func_optimizer, epoch, writer1):
    it = iter(dataloader)

    for i in range(train_pics):
        time_start = time.time()
        sample_dict = next(it)

        # prepare the data
        rgb_img = sample_dict['rgb_img']
        # rgb_img_scale2 = sample_dict['rgb_img_scale2']
        # rgb_img_scale4 = sample_dict['rgb_img_scale4']
        disp_gt = sample_dict['disp_gt']
        point_sample_index = sample_dict['point_sample_index']
        depth_sample_whole = sample_dict['depth_sample_whole'] # not-sampled points are zero, size as disp_gt
        index_list = sample_dict['index_list']

        rgb_img = rgb_img.to(device, torch.float32)
        # rgb_img_scale2 = rgb_img_scale2.to(device, torch.float32)
        # rgb_img_scale4 = rgb_img_scale4.to(device, torch.float32)
        disp_gt = disp_gt.to(device, torch.float32)
        depth_sample_whole = depth_sample_whole.to(device, torch.float32)
        index_list = index_list.squeeze(0).to(device, torch.float32)
        depth_sample_whole_copy = depth_sample_whole

        sample_mask = depth_sample_whole > 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

        feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution
        # feature_map_scale2 = feature_ex_net(rgb_img_scale2)
        # feature_map_scale4 = feature_ex_net(rgb_img_scale4)

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

        predict_sample_depth = torch.matmul(fc_concat, w)

        loss_1_smooth = loss_function(predict_sample_depth, depth_sample)

        # predict depth using W
        feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)
        # feature_map_scale2 = feature_map_scale2.reshape \
        #                     (1, feature_map_layers, (reso_scale2[0])*(reso_scale2[1])).permute(0, 2, 1)
        # feature_map_scale4 = feature_map_scale4.reshape \
        #                     (1, feature_map_layers, (reso_scale4[0])*(reso_scale4[1])).permute(0, 2, 1)

        # index_list = np.array([[i / resolution[0], j / resolution[1]] \
        #                     for i in range(resolution[0]) for j in range(resolution[1])]).transpose(1, 0)
        # index_list = torch.from_numpy(index_list).unsqueeze(0).to(device, torch.float32)

        # index_list_scale2 = np.array([[i / (reso_scale2[0]), j / (reso_scale2[1])] \
        #                     for i in range(reso_scale2[0]) for j in range(reso_scale2[1])]).transpose(1, 0)
        # index_list_scale2 = torch.from_numpy(index_list_scale2).unsqueeze(0).to(device, torch.float32)

        # index_list_scale4 = np.array([[i / (reso_scale4[0]), j / (reso_scale4[1])] \
        #                     for i in range(reso_scale4[0]) for j in range(reso_scale4[1])]).transpose(1, 0)
        # index_list_scale4 = torch.from_numpy(index_list_scale4).unsqueeze(0).to(device, torch.float32)

        fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()
        # fc_pic_in_scale2 = torch.cat((feature_map_scale2, \
        #                         position_encoding(index_list_scale2)), axis=2).squeeze()
        # fc_pic_in_scale4 = torch.cat((feature_map_scale4, \
        #                         position_encoding(index_list_scale4)), axis=2).squeeze()
        fc_pic_out = base_func(fc_pic_in)
        # fc_pic_out_scale2 = base_func(fc_pic_in_scale2)
        # fc_pic_out_scale4 = base_func(fc_pic_in_scale4)

        # add pose again
        fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)

        predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

        # multi-scale
        # predict_depth_scale2 = torch.matmul(fc_pic_out_scale2, w).reshape(1, 1, reso_scale2[0], reso_scale2[1])
        # predict_depth_scale4 = torch.matmul(fc_pic_out_scale4, w).reshape(1, 1, reso_scale4[0], reso_scale4[1])
        # final_predict = final_up_net(predict_depth, predict_depth_scale2, predict_depth_scale4)
        # predict_depth_scale2 = upsample_visual(predict_depth_scale2)
        # predict_depth_scale4 = upsample_visual(predict_depth_scale4)

        loss_2_smooth = loss_function(predict_depth, disp_gt)

        loss = lamda1 * loss_1_smooth + lamda2 * loss_2_smooth

        base_func_optimizer.zero_grad()
        feature_ex_net_optimizer.zero_grad()
        # final_up_net_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_func.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(feature_ex_net.parameters(), 10)
        # torch.nn.utils.clip_grad_norm_(final_up_net.parameters(), 10)
        base_func_optimizer.step()
        feature_ex_net_optimizer.step()
        # final_up_net_optimizer.step()

        disp_gt = disp_gt.squeeze().cpu().detach().numpy()
        predict_depth = predict_depth.squeeze().cpu().detach().numpy()
        # predict_depth_scale2 = predict_depth_scale2.squeeze().cpu().detach().numpy()
        # predict_depth_scale4 = predict_depth_scale4.squeeze().cpu().detach().numpy()
        depth_sample_whole_copy = depth_sample_whole_copy.squeeze().cpu().detach().numpy()

        writer1.add_scalar('loss', loss.item(), global_step=i+train_pics*epoch)
        writer1.add_scalar('loss_1', loss_1_smooth.item(), global_step=i+train_pics*epoch)
        writer1.add_scalar('loss_2', loss_2_smooth.item(), global_step=i+train_pics*epoch)

        if (i % 10 == 0):
            writer1.add_image('groundtruth and out', 
                np.concatenate((disp_gt, predict_depth, depth_sample_whole_copy), axis=1, out=None),
                # np.concatenate((disp_gt, predict_depth, predict_depth_scale2, predict_depth_scale4), axis=1, out=None),
                global_step=i+train_pics*epoch,
                dataformats='HW')
        
        print('Epoch:'+str(epoch)+' Pic:'+str(i)+" Processing a single picture using: %.2f s"%(time.time()-time_start))
        print("---------loss             : %.4f" % loss.item())
        print("---------loss_1           : %.4f" % loss_1_smooth.item())
        print("---------loss_2           : %.4f" % loss_2_smooth.item())


def eval_main_loop(eval_dataloader, feature_ex_net, base_func, epoch, writer1):
    it_eval = iter(eval_dataloader)
    loss_sum = 0.0
    for i in range(eval_pics):
        sample_dict_eval = next(it_eval)

        rgb_img = sample_dict_eval['rgb_img']
        # rgb_img_scale2 = sample_dict_eval['rgb_img_scale2']
        # rgb_img_scale4 = sample_dict_eval['rgb_img_scale4']
        disp_gt = sample_dict_eval['disp_gt']
        point_sample_index = sample_dict_eval['point_sample_index']
        depth_sample_whole = sample_dict_eval['depth_sample_whole'] # not-sampled points are zero, size as disp_gt
        index_list = sample_dict_eval['index_list']

        rgb_img = rgb_img.to(device, torch.float32)
        # rgb_img_scale2 = rgb_img_scale2.to(device, torch.float32)
        # rgb_img_scale4 = rgb_img_scale4.to(device, torch.float32)
        disp_gt = disp_gt.to(device, torch.float32)
        depth_sample_whole = depth_sample_whole.to(device, torch.float32)
        index_list = index_list.squeeze(0).to(device, torch.float32)
        # depth_sample_whole_copy = depth_sample_whole

        sample_mask = depth_sample_whole > 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

        feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution
        # feature_map_scale2 = feature_ex_net(rgb_img_scale2)
        # feature_map_scale4 = feature_ex_net(rgb_img_scale4)

        feature_map_sample = torch.masked_select(feature_map, sample_mask) # 256000
        feature_map_sample = feature_map_sample.reshape(1, feature_map_layers, sample_num).permute(0, 2, 1) # 1*1000*256

        # point_sample_index_norm ： 1*2*sample_num
        point_sample_index_norm = torch.from_numpy(np.array(point_sample_index, dtype=np.float32).transpose(1, 0))
        point_sample_index_norm = point_sample_index_norm.to(device, dtype=torch.float32) / \
                                    torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
        point_sample_index_norm = point_sample_index_norm.unsqueeze(0).to(device, dtype=torch.float32)

        fc_in = torch.cat((feature_map_sample, position_encoding(point_sample_index_norm)), dim=2) # 1*(256+40)*1000
        fc_in = fc_in.to(device).squeeze()

        fc_concat = base_func(fc_in) # 1 * sample_num * final_layers

        # add pose again
        fc_concat = torch.cat((fc_concat, position_encoding(point_sample_index_norm).squeeze()), dim=1)

        # caculate w
        w = least_squres(fc_concat, depth_sample)

        predict_sample_depth = torch.matmul(fc_concat, w)

        loss_1_smooth = loss_function(predict_sample_depth, depth_sample)

        # predict depth using W
        feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)
        # feature_map_scale2 = feature_map_scale2.reshape \
        #                     (1, feature_map_layers, (reso_scale2[0])*(reso_scale2[1])).permute(0, 2, 1)
        # feature_map_scale4 = feature_map_scale4.reshape \
        #                     (1, feature_map_layers, (reso_scale4[0])*(reso_scale4[1])).permute(0, 2, 1)

        # index_list = np.array([[i / resolution[0], j / resolution[1]] \
        #                     for i in range(resolution[0]) for j in range(resolution[1])]).transpose(1, 0)
        # index_list = torch.from_numpy(index_list).unsqueeze(0).to(device, torch.float32)

        # index_list_scale2 = np.array([[i / (reso_scale2[0]), j / (reso_scale2[1])] \
        #                     for i in range(reso_scale2[0]) for j in range(reso_scale2[1])]).transpose(1, 0)
        # index_list_scale2 = torch.from_numpy(index_list_scale2).unsqueeze(0).to(device, torch.float32)

        # index_list_scale4 = np.array([[i / (reso_scale4[0]), j / (reso_scale4[1])] \
        #                     for i in range(reso_scale4[0]) for j in range(reso_scale4[1])]).transpose(1, 0)
        # index_list_scale4 = torch.from_numpy(index_list_scale4).unsqueeze(0).to(device, torch.float32)
        
        fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()
        # fc_pic_in_scale2 = torch.cat((feature_map_scale2, \
        #                         position_encoding(index_list_scale2)), axis=2).squeeze()
        # fc_pic_in_scale4 = torch.cat((feature_map_scale4, \
        #                         position_encoding(index_list_scale4)), axis=2).squeeze()
        fc_pic_out = base_func(fc_pic_in)
        # fc_pic_out_scale2 = base_func(fc_pic_in_scale2)
        # fc_pic_out_scale4 = base_func(fc_pic_in_scale4)

        # add pose again
        fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)
        
        predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

        # multi-scale
        # predict_depth_scale2 = torch.matmul(fc_pic_out_scale2, w).reshape(1, 1, reso_scale2[0], reso_scale2[1])
        # predict_depth_scale4 = torch.matmul(fc_pic_out_scale4, w).reshape(1, 1, reso_scale4[0], reso_scale4[1])
        # final_predict = final_up_net(predict_depth, predict_depth_scale2, predict_depth_scale4)

        loss_2_smooth = loss_function(predict_depth, disp_gt)

        loss = lamda1 * loss_1_smooth + lamda2 * loss_2_smooth

        loss_sum += loss.item()
    print("========================================")
    print("Evaluation loss is: ", loss_sum)
    writer1.add_scalar('eval_loss', loss_sum, global_step=epoch)


def train_main():
    base_func = fc.FC_basefunction(final_layer, feature_map_layers+20)
    base_func.load_state_dict(torch.load("./checkpoints/base_bugfixed_baseline.tar"))
    base_func = base_func.to(device)

    feature_ex_net = fc.DRPNet()
    feature_ex_net.load_state_dict(torch.load("./checkpoints/unet_bugfixed_baseline.tar"))
    feature_ex_net = feature_ex_net.to(device)

    # final_up_net = fc.FinalUpsample(resolution, reso_scale2, reso_scale4)
    # final_up_net.load_state_dict(torch.load("./checkpoints/unet_add_grad_loss_80.tar"))
    # final_up_net = final_up_net.to(device)

    base_func_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.00001, weight_decay=5e-4)
    feature_ex_net_optimizer = torch.optim.Adam(feature_ex_net.parameters(), lr=0.00001, weight_decay=5e-4)
    # final_up_net_optimizer = torch.optim.Adam(final_up_net.parameters(), lr=0.00001, weight_decay=5e-4)

    writer1 = SummaryWriter(log_dir='runs/' + run_file_name)

    # flying_dataset = fly.FlyingDataset(filelistpath="./data/flying3d_train.txt",
    #                                     transform=transforms.Compose([
    #                                         fly.Rescale((resolution, resolution))
    #                                     ]))
    # dataloader = DataLoader(flying_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
    # it = iter(dataloader)

    nyu_dataset = nyu.NYUDataset(filelistpath="./data/NYUv2_train.txt",
                                 transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                           size_scale2=reso_scale2,
                                                                           size_scale4=reso_scale4), 
                                                               nyu.ToTensor(),
                                                               nyu.SamplePoint(reso=resolution, 
                                                                               sample_=sample_num)]))
    dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)


    nyu_eval_dataset = nyu.NYUDataset(filelistpath="./data/NYUv2_eval.txt",
                                      transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                                size_scale2=reso_scale2,
                                                                                size_scale4=reso_scale4), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=resolution, 
                                                                                    sample_=sample_num)]))
    eval_dataloader = DataLoader(nyu_eval_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    for epoch in range(total_epochs):
        base_func.train()
        feature_ex_net.train()
        # final_up_net.train()
        train_main_loop(dataloader, feature_ex_net, base_func, feature_ex_net_optimizer, \
                        base_func_optimizer, epoch, writer1)

        if eval_bool:
            print("========================================")
            print("Start evaluating the net......")

            base_func.eval()
            feature_ex_net.eval()
            # final_up_net.eval()
            eval_main_loop(eval_dataloader, feature_ex_net, base_func, epoch, writer1)
        
        if (True and (epoch % 10 == 0) and (epoch != 0)):
            savefilename = './checkpoints/base_'  + run_file_name + "_" + str(epoch)+'.tar'
            torch.save(base_func.state_dict(), savefilename)
            savefilename = './checkpoints/unet_' + run_file_name + "_" + str(epoch)+'.tar'
            torch.save(feature_ex_net.state_dict(), savefilename)
            # savefilename = './checkpoints/upsample_256_1000_no_multiscale_'+str(epoch)+'.tar'
            # torch.save(final_up_net.state_dict(), savefilename)


if __name__ == '__main__':
    train_main()
