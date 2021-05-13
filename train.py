#!/home/nickle/miniconda3/envs/siren3d/bin/python

import torch
import time
import os
import cv2
import math
import cmapy
import scipy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.transforms.transforms import RandomCrop
import matplotlib.colors as colors

from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
shuffle_bool = True
eval_bool = True
train_pics = 8000#47584
eval_pics = 654
total_epochs = 20000
sample_num = 1000
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (228, 304)
raw_resolution = (480, 640)
reso_scale2 = (240, 320)
reso_scale4 = (120, 160)
reso_scale8 = (64, 80)
lamda1 = 1 # loss_1
lamda2 = 1 # loss_2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_file_name = "test_shuffle"
use_dxdy = True
data_gain_bool = True

# define loss function
# loss_function = torch.nn.SmoothL1Loss(beta=0.001)
# loss_function = torch.nn.L1Loss()
# l2_loss_function = torch.nn.MSELoss()

class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss_function = torch.nn.L1Loss()
    
    def forward(self, gt, predict, mask=None):
        if (mask==None):
            loss = self.loss_function(gt, predict)    
        else:
            loss = self.loss_function(gt[mask], predict[mask])

        return loss

upsample_visual = torch.nn.Upsample(size=raw_resolution, mode="bilinear", align_corners=True)
cm = plt.get_cmap('plasma')


def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


class Sobel(torch.nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = torch.nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = -out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

get_gradient = Sobel().to(device)
cos = torch.nn.CosineSimilarity(dim=1, eps=0)


def gradloss(disp_gt, predict_depth):
    depth_grad = get_gradient(disp_gt)
    output_grad = get_gradient(predict_depth)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(disp_gt)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(disp_gt)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(disp_gt)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(disp_gt)

    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1).mean()

    ones = torch.ones_like(disp_gt).float().to(device)
    ones = torch.autograd.Variable(ones)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

    return (loss_dx + loss_dy + loss_normal)


def position_encoding(tensor):
    x, y = torch.split(tensor, 1, dim=1)
    # encoded = torch.cat((torch.sin((2**0)*math.pi*x), torch.cos((2**0)*math.pi*x),
    #                     torch.sin((2**0)*math.pi*y), torch.cos((2**0)*math.pi*y)), dim=1)
    encoded = torch.cat((x, x, y, y), dim=1)
    for i in range(9):
        encoded = torch.cat((encoded, 
                            torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
                            torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

    return encoded


def evaluation(gt, predict_depth, t_valid=0.001):
    with torch.no_grad():
        pred_inv = 1.0 / (predict_depth + 1e-8)
        gt_inv = 1.0 / (gt + 1e-8)

        # For numerical stability
        mask = gt > t_valid
        num_valid = mask.sum()

        pred = predict_depth[mask]
        gt = gt[mask]

        pred_inv = pred_inv[mask]
        gt_inv = gt_inv[mask]

        pred_inv[pred <= t_valid] = 0.0
        gt_inv[gt <= t_valid] = 0.0

        # RMSE / MAE
        diff = pred - gt
        diff_abs = torch.abs(diff)
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

        mae = diff_abs.sum() / (num_valid + 1e-8)

        # iRMSE / iMAE
        diff_inv = pred_inv - gt_inv
        diff_inv_abs = torch.abs(diff_inv)
        diff_inv_sqr = torch.pow(diff_inv, 2)

        irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
        irmse = torch.sqrt(irmse)

        imae = diff_inv_abs.sum() / (num_valid + 1e-8)

        # Rel
        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)

        # delta
        r1 = gt / (pred + 1e-8)
        r2 = pred / (gt + 1e-8)
        ratio = torch.max(r1, r2)

        del_1 = (ratio < 1.25).type_as(ratio)
        del_2 = (ratio < 1.25**2).type_as(ratio)
        del_3 = (ratio < 1.25**3).type_as(ratio)

        del_1 = del_1.sum() / (num_valid + 1e-8)
        del_2 = del_2.sum() / (num_valid + 1e-8)
        del_3 = del_3.sum() / (num_valid + 1e-8)

        return {'rmse': rmse, 
                'mae': mae, 
                'irmse': irmse, 
                'imae': imae, 
                'rel': rel, 
                'del_1': del_1, 
                'del_2': del_2, 
                'del_3': del_3}


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
        try:
            sample_dict = next(it)
        except:
            continue

        # prepare the data
        rgb_img = sample_dict['rgb_img']
        disp_gt = sample_dict['disp_gt']
        point_sample_index = sample_dict['point_sample_index']
        depth_sample_whole = sample_dict['depth_sample_whole'] # not-sampled points are zero, size as disp_gt
        index_list = sample_dict['index_list']

        rgb_img = rgb_img.to(device, torch.float32)
        disp_gt = disp_gt.to(device, torch.float32)
        depth_sample_whole = depth_sample_whole.to(device, torch.float32)
        index_list = index_list.to(device, torch.float32)
        point_sample_index = point_sample_index.to(device, torch.float32)
        depth_sample_whole_copy = depth_sample_whole

        disp_gt = torch.autograd.Variable(disp_gt)

        sample_mask = depth_sample_whole >= 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

        feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution

        feature_map_sample = torch.masked_select(feature_map, sample_mask) # 256000
        feature_map_sample = feature_map_sample.reshape(1, feature_map_layers, sample_num).permute(0, 2, 1) # 1*1000*256

        point_sample_index_norm = point_sample_index / torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)

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

        index_list = index_list / torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
        fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()
        fc_pic_out = base_func(fc_pic_in)

        # add pose again
        fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)

        predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

        mask = disp_gt > 0
        loss_2_smooth = loss_function(predict_depth[mask], disp_gt[mask])

        # add gradient loss
        depth_grad = get_gradient(disp_gt)
        output_grad = get_gradient(predict_depth)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(disp_gt)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(disp_gt)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(disp_gt)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(disp_gt)

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1).mean()

        ones = torch.ones_like(disp_gt).float().to(device)
        ones = torch.autograd.Variable(ones)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        loss = lamda1 * loss_1_smooth + lamda2 * loss_2_smooth + (loss_dx + loss_dy + loss_normal)

        base_func_optimizer.zero_grad()
        feature_ex_net_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_func.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(feature_ex_net.parameters(), 10)
        base_func_optimizer.step()
        feature_ex_net_optimizer.step()

        # nyudepthv2 visualization
        disp_gt = disp_gt / 10
        disp_gt = disp_gt.squeeze().detach().cpu().numpy()
        disp_gt = (255.0*cm(disp_gt)).astype('uint8')

        predict_depth = predict_depth / 10
        predict_depth = predict_depth.squeeze().detach().cpu().numpy()
        predict_depth = (255.0*cm(predict_depth)).astype('uint8')

        depth_sample_whole_copy = depth_sample_whole_copy / 10
        depth_sample_whole_copy = depth_sample_whole_copy.squeeze().detach().cpu().numpy()
        depth_sample_whole_copy = (255.0*cm(depth_sample_whole_copy)).astype('uint8')

        writer1.add_scalar('loss', loss.item(), global_step=i+train_pics*epoch)
        writer1.add_scalar('loss_1', loss_1_smooth.item(), global_step=i+train_pics*epoch)
        writer1.add_scalar('loss_2', loss_2_smooth.item(), global_step=i+train_pics*epoch)

        if (i % 10 == 0):
            writer1.add_image('groundtruth and out', 
                np.concatenate((disp_gt, predict_depth, depth_sample_whole_copy), axis=1),
                # np.concatenate((disp_gt, predict_depth, depth_sample_whole_copy), axis=1, out=None),
                # np.concatenate((disp_gt, predict_depth, predict_depth_scale2, predict_depth_scale4), axis=1, out=None),
                global_step=i+train_pics*epoch,
                dataformats='HWC')

        print('Epoch:'+str(epoch)+' Pic:'+str(i)+" Processing a single picture using: %.2f s"%(time.time()-time_start))
        print("---------loss             : %.4f" % loss.item())
        print("---------loss_1           : %.4f" % loss_1_smooth.item())
        print("---------loss_2           : %.4f" % loss_2_smooth.item())


def eval_main_loop(eval_dataloader, feature_ex_net, base_func, epoch, writer1):
    it_eval = iter(eval_dataloader)
    pbar = tqdm(total=eval_pics)

    rmse_total = 0.0
    mae_total = 0.0
    irmse_total = 0.0
    imae_total = 0.0
    rel_total = 0.0
    del_1_total = 0.0
    del_2_total = 0.0
    del_3_total = 0.0
    for i in range(eval_pics):
        with torch.no_grad():
            try:
                sample_dict_eval = next(it_eval)
            except:
                continue

            rgb_img = sample_dict_eval['rgb_img']
            disp_gt = sample_dict_eval['disp_gt']
            point_sample_index = sample_dict_eval['point_sample_index']
            depth_sample_whole = sample_dict_eval['depth_sample_whole'] # not-sampled points are zero, size as disp_gt
            index_list = sample_dict_eval['index_list']

            rgb_img = rgb_img.to(device, torch.float32)
            disp_gt = disp_gt.to(device, torch.float32)
            depth_sample_whole = depth_sample_whole.to(device, torch.float32)
            index_list = index_list.to(device, torch.float32)
            point_sample_index = point_sample_index.to(device, torch.float32)

            sample_mask = depth_sample_whole > 0
            depth_sample = torch.masked_select(depth_sample_whole, sample_mask).reshape(sample_num, 1)

            feature_map = feature_ex_net(rgb_img) # 1*256*resolution*resolution

            feature_map_sample = torch.masked_select(feature_map, sample_mask) # 256000
            feature_map_sample = feature_map_sample.reshape(1, feature_map_layers, sample_num).permute(0, 2, 1) # 1*1000*256

            # point_sample_index_norm ： 1*2*sample_num
            point_sample_index_norm = point_sample_index / torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)

            fc_in = torch.cat((feature_map_sample, position_encoding(point_sample_index_norm)), dim=2) # 1*(256+40)*1000
            fc_in = fc_in.to(device).squeeze()

            fc_concat = base_func(fc_in) # 1 * sample_num * final_layers

            # add pose again
            fc_concat = torch.cat((fc_concat, position_encoding(point_sample_index_norm).squeeze()), dim=1)
            # caculate w
            w = least_squres(fc_concat, depth_sample)
            # predict depth using W
            feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)
            
            index_list = index_list / torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
            fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()
            fc_pic_out = base_func(fc_pic_in)

            # add pose again
            fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)
            
            predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

            loss_dict = evaluation(disp_gt, predict_depth, 0.0001)

            rmse = loss_dict['rmse'].item()
            mae = loss_dict["mae"].item()
            irmse = loss_dict["irmse"].item()
            imae = loss_dict["imae"].item()
            rel = loss_dict["rel"].item()
            del_1 = loss_dict["del_1"].item()
            del_2 = loss_dict["del_2"].item()
            del_3 = loss_dict["del_3"].item()

            rmse_total += rmse
            mae_total += mae
            irmse_total += irmse
            imae_total += imae
            rel_total += rel
            del_1_total += del_1
            del_2_total += del_2
            del_3_total += del_3

            error_str = 'RMSE= {:.3f} | REL= {:.3f} | D1= {:.3f} | D2= {:.3f} | D3= {:.3f}'.format(
                        rmse, rel, del_1, del_2, del_3)
            pbar.set_description(error_str)
            pbar.update(1)

    rmse_total /= eval_pics
    mae_total /= eval_pics
    irmse_total /= eval_pics
    imae_total /= eval_pics
    rel_total /= eval_pics
    del_1_total /= eval_pics
    del_2_total /= eval_pics
    del_3_total /= eval_pics
    print("========================================")
    print("RMSE is: ", rmse_total)
    print("REL is:", rel_total)
    writer1.add_scalar('RMSE', rmse_total, global_step=epoch)
    writer1.add_scalar('REL', rel_total, global_step=epoch)

    return rmse_total, rel_total


def train_main():
    base_func = fc.FC_basefunction(final_layer, feature_map_layers+20)
    base_func.load_state_dict(torch.load("./checkpoints/base_gain_with_gradient_loss_rmse_0.16701324204615223_rel_0.04016858216842198_0.tar"))
    base_func = base_func.to(device)

    feature_ex_net = fc.DRPNet()
    feature_ex_net.load_state_dict(torch.load("./checkpoints/unet_gain_with_gradient_loss_rmse_0.16701324204615223_rel_0.04016858216842198_0.tar"))
    feature_ex_net = feature_ex_net.to(device)

    # base_func_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.000001, weight_decay=5e-4)
    # feature_ex_net_optimizer = torch.optim.Adam(feature_ex_net.parameters(), lr=0.00001, weight_decay=5e-4)

    base_func_optimizer = torch.optim.SGD(base_func.parameters(), lr=0.000001, momentum=0.9)
    feature_ex_net_optimizer = torch.optim.SGD(feature_ex_net.parameters(), lr=0.00001, momentum=0.9)

    writer1 = SummaryWriter(log_dir='runs/' + run_file_name)

    # flying_dataset = fly.FlyingDataset(filelistpath="./data/flying3d_train.txt",
    #                                     transform=transforms.Compose([
    #                                         fly.Rescale((resolution, resolution))
    #                                     ]))
    # dataloader = DataLoader(flying_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
    # it = iter(dataloader)

    # nyu_dataset = nyu.NYUDataset(filelistpath="./data/NYUv2_train.txt",
    #                              transform=transforms.Compose([nyu.Rescale(resolution, 
    #                                                                        size_scale2=reso_scale2,
    #                                                                        size_scale4=reso_scale4), 
    #                                                            nyu.ToTensor(),
    #                                                            nyu.SamplePoint(reso=resolution, 
    #                                                                            sample_=sample_num)]))
    # dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)


    nyu_eval_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                      transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                                data_gain=False), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=resolution, 
                                                                                    sample_=sample_num)]))
    eval_dataloader = DataLoader(nyu_eval_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    nyu_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_train.txt",
                                    transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                              data_gain=data_gain_bool), 
                                                                nyu.ToTensor(),
                                                                nyu.SamplePoint(reso=resolution, 
                                                                               sample_=sample_num)]))
    dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    for epoch in range(total_epochs):
        base_func.train()
        feature_ex_net.train()
        train_main_loop(dataloader, feature_ex_net, base_func, feature_ex_net_optimizer, \
                        base_func_optimizer, epoch, writer1)

        if eval_bool:
            print("========================================")
            print("Start evaluating the net......")

            base_func.eval()
            feature_ex_net.eval()
            rmse_total, rel_total = eval_main_loop(eval_dataloader, feature_ex_net, base_func, epoch, writer1)
        
        # if (True and (epoch % 10 == 0) and (epoch != 0)):
        savefilename = './checkpoints/base_'  + run_file_name + "_rmse_" \
                        + str(rmse_total) + "_rel_" + str(rel_total) + "_" + str(epoch)+'.tar'
        torch.save(base_func.state_dict(), savefilename)
        savefilename = './checkpoints/unet_'  + run_file_name + "_rmse_" \
                        + str(rmse_total) + "_rel_" + str(rel_total) + "_" + str(epoch)+'.tar'
        torch.save(feature_ex_net.state_dict(), savefilename)


def train_multiscale():
    learning_rate = 1e-5
    net = fc.PyramidDepthEstimation(reso_scale2, reso_scale4, reso_scale8, raw_resolution, sample_num)
    net = net.to(device)
    net.load_state_dict(torch.load("./checkpoints/unet_baseline.tar"))
    # net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    net_optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    nyu_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_train.txt",
                                    transform=transforms.Compose([nyu.Rescale_Multiscale \
                                                                 (raw_resolution, reso_scale2, reso_scale4, reso_scale8), 
                                                            nyu.ToTensor_Multiscale(),
                                                            nyu.SamplePoint_Multiscale \
                                                                (reso_scale4, sample_num)]))
    dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    # nyu_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_train.txt",
    #                                   transform=transforms.Compose([nyu.Rescale(reso_scale2, 
    #                                                                             data_gain=False), 
    #                                                                 nyu.ToTensor(),
    #                                                                 nyu.SamplePoint(reso=reso_scale2, 
    #                                                                                 sample_=sample_num)]))
    # dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    writer = SummaryWriter(log_dir='runs/' + run_file_name)

    lossfunction = LossFunction()

    for epoch in range(total_epochs):
        net.train()
        it = iter(dataloader)

        # if (epoch > 3):
        #     learning_rate = 1e-5
        # if (epoch > 5):
        #     learning_rate = 1e-6
        # if (epoch > 8):
        #     learning_rate = 1e-7
        # net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

        for i in range(train_pics):
            time_1 = time.time()
            try:
                sample_dict = next(it)
            except:
                continue
            rgb_scale2 = sample_dict['rgb_scale2'].to(device)
            rgb_scale4 = sample_dict['rgb_scale4'].to(device)
            # rgb_scale8 = sample_dict['rgb_scale8'].to(device)
            # depth_scale2 = sample_dict['depth_scale2'].to(device)
            # depth_scale4 = sample_dict['depth_scale4'].to(device)
            # depth_scale8 = sample_dict['depth_scale8'].to(device)
            depth_sample_whole = sample_dict['depth_sample_whole'].to(device)
            disp_gt_raw = sample_dict['disp_gt_raw'].to(device)

            disp_gt_raw = torch.autograd.Variable(disp_gt_raw)
            # depth_scale4 = torch.autograd.Variable(depth_scale4)
            # depth_scale8 = torch.autograd.Variable(depth_scale8)

            time_2 = time.time()

            # loss1, predict_depth, pre_scale4, pre_scale8 = net(rgb_scale8, rgb_scale4, rgb_scale2, depth_sample_whole)
            loss1, delta_h_loss, delta_w_loss, depth_finetune_1, depth_finetune_2, predict_init_depth = \
                net(rgb_scale4, rgb_scale2, depth_sample_whole, use_dxdy=use_dxdy)

            predict_depth_kid = upsample_visual(depth_finetune_1)
            predict_depth_final = upsample_visual(depth_finetune_2)
            predict_init_depth = upsample_visual(predict_init_depth)

            time_3 = time.time()

            loss2 = lossfunction(disp_gt_raw, predict_init_depth)
            loss3 = lossfunction(disp_gt_raw, predict_depth_kid)
            loss4 = lossfunction(disp_gt_raw, predict_depth_final)

            grad_loss = gradloss(disp_gt_raw, predict_init_depth) + gradloss(disp_gt_raw, predict_depth_kid) \
                        + gradloss(disp_gt_raw, predict_depth_final)

            # loss3 = lossfunction(depth_scale4, pre_scale4)
            # loss4 = lossfunction(depth_scale8, pre_scale8)

            # loss = loss1 + loss4 + loss3 + loss2 + gradloss2 + gradloss3 + gradloss4
            if (delta_h_loss != None):
                loss = loss1 + 2*loss2 + loss3 + loss4 + grad_loss + delta_h_loss + delta_w_loss
            else:
                loss = loss1 + 2*loss2 + loss3 + loss4 + grad_loss

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            net_optimizer.step()

            print(run_file_name + ' Epoch:'+str(epoch)+' Pic:'+str(i)+" Using: %.2f s"%(time.time()-time_1))
            print("---------loss   : %.4f" % loss.item())

            writer.add_scalar('loss', loss.item(), global_step=i+train_pics*epoch)

            # nyudepthv2 visualization
            # pre_scale4 = upsample_visual(pre_scale4)
            # pre_scale8 = upsample_visual(pre_scale8)
            # depth_sample_whole = upsample_visual(depth_sample_whole)
            
            # depth_scale2 = depth_scale2 / 10
            # depth_scale2 = depth_scale2.squeeze().detach().cpu().numpy()
            # depth_scale2 = (255.0*cm(depth_scale2)).astype('uint8')

            disp_gt_raw = disp_gt_raw / 10
            disp_gt_raw = disp_gt_raw.squeeze().detach().cpu().numpy()
            disp_gt_raw = (255.0*cm(disp_gt_raw)).astype('uint8')

            # depth_scale8 = depth_scale8 / 10
            # depth_scale8 = depth_scale8.squeeze().detach().cpu().numpy()
            # depth_scale8 = (255.0*cm(depth_scale8)).astype('uint8')

            # pre_scale8 = pre_scale8 / 10
            # pre_scale8 = pre_scale8.squeeze().detach().cpu().numpy()
            # pre_scale8 = (255.0*cm(pre_scale8)).astype('uint8')

            # pre_scale4 = pre_scale4 / 10
            # pre_scale4 = pre_scale4.squeeze().detach().cpu().numpy()
            # pre_scale4 = (255.0*cm(pre_scale4)).astype('uint8')

            predict_depth_kid = predict_depth_kid / 10
            predict_depth_kid = predict_depth_kid.squeeze().detach().cpu().numpy()
            predict_depth_kid = (255.0*cm(predict_depth_kid)).astype('uint8')

            predict_depth_final = predict_depth_final / 10
            predict_depth_final = predict_depth_final.squeeze().detach().cpu().numpy()
            predict_depth_final = (255.0*cm(predict_depth_final)).astype('uint8')

            predict_init_depth = predict_init_depth / 10
            predict_init_depth = predict_init_depth.squeeze().detach().cpu().numpy()
            predict_init_depth = (255.0*cm(predict_init_depth)).astype('uint8')

            # depth_sample_whole = depth_sample_whole / 10
            # depth_sample_whole = depth_sample_whole.squeeze().detach().cpu().numpy()
            # depth_sample_whole = (255.0*cm(depth_sample_whole)).astype('uint8')

            # rgb_scale2 = rgb_scale2.squeeze().permute(1, 2, 0).detach().cpu().numpy()

            if (i % 30 == 0):
                writer.add_image('groundtruth and out', 
                    # np.concatenate((depth_scale2, depth_sample_whole, predict_depth, pre_scale4, pre_scale8), axis=1),
                    np.concatenate((disp_gt_raw, predict_init_depth, predict_depth_kid, predict_depth_final), axis=1),
                    global_step=i+train_pics*epoch,
                    dataformats='HWC')
                # writer.add_image('RGB', 
                #     rgb_scale2,
                #     global_step=i+train_pics*epoch,
                #     dataformats='HWC')

        # evaluation
        nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                    transform=transforms.Compose([nyu.Rescale_Multiscale \
                                                                 (raw_resolution, reso_scale2, reso_scale4, reso_scale8), 
                                                            nyu.ToTensor_Multiscale(),
                                                            nyu.SamplePoint_Multiscale \
                                                                (reso_scale4, sample_num)]))
        dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

        # nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
        #                                      transform=transforms.Compose([nyu.Rescale(reso_scale2, 
        #                                                                                data_gain=False), 
        #                                                             nyu.ToTensor(),
        #                                                             nyu.SamplePoint(reso=reso_scale2, 
        #                                                                             sample_=sample_num)]))
        # dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

        it = iter(dataloader_eval)

        pbar = tqdm(total=eval_pics)

        rmse_total = 0.0
        mae_total = 0.0
        irmse_total = 0.0
        imae_total = 0.0
        rel_total = 0.0
        del_1_total = 0.0
        del_2_total = 0.0
        del_3_total = 0.0

        for i in range(eval_pics):
            with torch.no_grad():
                try:
                    sample_dict = next(it)
                except:
                    continue

                rgb_scale2 = sample_dict['rgb_scale2'].to(device)
                rgb_scale4 = sample_dict['rgb_scale4'].to(device)
                # rgb_scale8 = sample_dict['rgb_scale8'].to(device)
                disp_gt_raw = sample_dict['disp_gt_raw'].to(device)
                depth_sample_whole = sample_dict['depth_sample_whole'].to(device)

                # loss1, predict_depth, pre_scale4, pre_scale8 = net(rgb_scale8, rgb_scale4, rgb_scale2, depth_sample_whole)
                _, __, ___, depth_finetune_1, depth_finetune_2, predict_init_depth = net(rgb_scale4, rgb_scale2, depth_sample_whole)
                predict_depth = upsample_visual(depth_finetune_2)

                loss_dict = evaluation(disp_gt_raw, predict_depth)

                rmse = loss_dict['rmse'].item()
                mae = loss_dict["mae"].item()
                irmse = loss_dict["irmse"].item()
                imae = loss_dict["imae"].item()
                rel = loss_dict["rel"].item()
                del_1 = loss_dict["del_1"].item()
                del_2 = loss_dict["del_2"].item()
                del_3 = loss_dict["del_3"].item()

                rmse_total += rmse
                mae_total += mae
                irmse_total += irmse
                imae_total += imae
                rel_total += rel
                del_1_total += del_1
                del_2_total += del_2
                del_3_total += del_3

                error_str = 'RMSE= {:.3f} | REL= {:.3f} | D1= {:.3f} | D2= {:.3f} | D3= {:.3f}'.format(
                            rmse, rel, del_1, del_2, del_3)
                pbar.set_description(error_str)
                pbar.update(1)

        rmse_total = rmse_total / eval_pics
        rel_total = rel_total / eval_pics
        del_1_total = del_1_total / eval_pics
        del_2_total = del_2_total / eval_pics
        del_3_total = del_3_total / eval_pics
        print("rmse is: ", rmse_total)
        print("rel is: ", rel_total)
        print("d1 is: ", del_1_total)
        print("d2 is: ", del_2_total)
        print("d3 is: ", del_3_total)

        writer.add_scalar('RMSE', rmse_total, global_step=epoch)
        writer.add_scalar('REL', rel_total, global_step=epoch)
        writer.add_scalar('DELTA1', del_1_total, global_step=epoch)
        writer.add_scalar('DELTA2', del_2_total, global_step=epoch)
        writer.add_scalar('DELTA3', del_3_total, global_step=epoch)

        savefilename = './checkpoints/' + run_file_name + '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                        + "_epoch_" + str(epoch)+'.tar'
        torch.save(net.state_dict(), savefilename)


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x

def to_uint8(x):
    return (255. * x).astype(np.uint8)

def train_update():
    net_learning_rate = 1e-5
    finetune_learning_rate = 1e-5
    conv_init_learning_rate = 1e-5
    net = fc.Siren3dEstimation(sample_num)
    finetune = fc.Finetune()
    convinit = fc.ConvInit()
    net = net.to(device)
    finetune = finetune.to(device)
    convinit = convinit.to(device)
    net.load_state_dict(torch.load("./checkpoints/shuabang/baseline_net_rmse_0.19360252093261718_rel_0.05172041724991361_epoch_17.tar"))
    finetune.load_state_dict(torch.load("./checkpoints/shuabang/baseline_finetune_rmse_0.19360252093261718_rel_0.05172041724991361_epoch_17.tar"))
    # convinit.load_state_dict(torch.load("./checkpoints/"))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=net_learning_rate, weight_decay=5e-4)
    finetune_optimizer = torch.optim.Adam(finetune.parameters(), lr=finetune_learning_rate, weight_decay=5e-4)
    convinit_optimizer = torch.optim.Adam(convinit.parameters(), lr=conv_init_learning_rate, weight_decay=5e-4)

    nyu_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_train.txt",
                                    transform=transforms.Compose([nyu.Rescale_update(train=True),
                                                                nyu.ToTensor_update(),
                                                                nyu.SamplePoint_update(sample_num)]))
    dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)
    
    lossfunction = LossFunction().to(device)
    writer = SummaryWriter(log_dir='runs/' + run_file_name)

    sobel = Sobel().to(device)

    for epoch in range(total_epochs):
        net.train()
        finetune.train()
        # convinit.train()
        it = iter(dataloader)

        for i in range(train_pics):
            time_1 = time.time()
            try:
                sample_dict = next(it)
            except:
                continue

            depth = sample_dict['depth'].to(device)
            rgb = sample_dict['rgb'].to(device)
            index_list = sample_dict['index_list'].to(device)
            depth_sample_whole = sample_dict['depth_sample_whole'].to(device)
            gt_gradient = sample_dict['gt_gradient'].to(device)

            index_list = index_list / torch.tensor([resolution[0]-1, resolution[1]-1]).reshape(1, 2, 1, 1).to(device)
            # index_list = index_list * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)

            filter_mask = depth > 0
            if (filter_mask.sum() < 67000):
                print("Bad Data!!!!!!!!!")
                continue

            index_list_copy = index_list.clone()

            time_2 = time.time()

            loss1, predict_init_depth = net(rgb, depth_sample_whole, index_list, index_list_copy)
            # conv_init_depth = convinit(predict_init_depth)
            finetune_depth = finetune(predict_init_depth, rgb)

            time_3 = time.time()

            # predict_init_depth_copy = predict_init_depth.clone()

            # grad_outputs = torch.ones_like(predict_init_depth).to(device)
            # pre_grad = torch.autograd.grad(predict_init_depth, index_list_copy, grad_outputs=grad_outputs, create_graph=True)[0]

            # pre_laplace = divergence(pre_grad, index_list)
 
            mask = depth > 0
            loss2 = lossfunction(depth, predict_init_depth, mask=mask)
            # loss_convinit = lossfunction(depth, conv_init_depth)
            loss3 = lossfunction(depth, finetune_depth, mask=mask)

            gt_depth_resize_sobel = sobel(depth)
            pre_init_sobel = sobel(predict_init_depth)
            # conv_init_sobel = sobel(conv_init_depth)
            finetune_sobel = sobel(finetune_depth)
            grad_loss_init = lossfunction(gt_depth_resize_sobel, pre_init_sobel)
            # grad_loss_convinit = lossfunction(gt_depth_resize_sobel, conv_init_sobel)
            grad_loss_finetune = lossfunction(gt_depth_resize_sobel, finetune_sobel)

            # mask = mask.squeeze(0).reshape(1, -1, 1)
            # mask = torch.cat((mask, mask), dim=2)
            # pre_grad = pre_grad.permute(0, 2, 1)
            # grad_loss = lossfunction(gt_gradient, pre_grad, mask=mask)

            loss = loss1 + loss2 + grad_loss_init + loss3 + grad_loss_finetune
                                #  + loss_convinit + grad_loss_convinit
                                 # + 0.05*grad_loss

            net.zero_grad()
            finetune.zero_grad()
            # convinit.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(finetune.parameters(), 10)
            # torch.nn.utils.clip_grad_norm_(convinit.parameters(), 10)
            net_optimizer.step()
            finetune_optimizer.step()
            # convinit_optimizer.step()

            time_4 = time.time()

            print(run_file_name + ' Epoch:'+str(epoch)+' Pic:'+str(i)+" Using: %.2f s"%(time.time()-time_1))
            print("---------loss   : %.4f" % loss.item())

            writer.add_scalar('loss', loss.item(), global_step=i+train_pics*epoch)

            # visualization
            gt_visual_init_sobel = torch.cat((gt_depth_resize_sobel[:, 0, :, :].reshape(-1, 1), 
                                           gt_depth_resize_sobel[:, 1, :, :].reshape(-1, 1)), dim=-1).unsqueeze(0)
            pre_visual_init_sobel = torch.cat((pre_init_sobel[:, 0, :, :].reshape(-1, 1), 
                                           pre_init_sobel[:, 1, :, :].reshape(-1, 1)), dim=-1).unsqueeze(0)
            gt_visual_init_sobel_img = grads2img(lin2img(gt_visual_init_sobel, image_resolution=resolution)).permute(1, 2, 0).numpy()
            pre_visual_init_sobel_img = grads2img(lin2img(pre_visual_init_sobel, image_resolution=resolution)).permute(1, 2, 0).numpy()

            depth = depth / 10
            depth = depth.squeeze().detach().cpu().numpy()
            depth = (255.0*cm(depth)).astype('uint8')

            predict_init_depth = predict_init_depth / 10
            predict_init_depth = predict_init_depth.squeeze().detach().cpu().numpy()
            predict_init_depth = (255.0*cm(predict_init_depth)).astype('uint8')

            # conv_init_depth = conv_init_depth / 10
            # conv_init_depth = conv_init_depth.squeeze().detach().cpu().numpy()
            # conv_init_depth = (255.0*cm(conv_init_depth)).astype('uint8')

            finetune_depth = finetune_depth / 10
            finetune_depth = finetune_depth.squeeze().detach().cpu().numpy()
            finetune_depth = (255.0*cm(finetune_depth)).astype('uint8')
            ################################################################

            # gt_gradient_img = grads2img(lin2img(gt_gradient, image_resolution=resolution)).permute(1, 2, 0).numpy()
            # pre_gradient_img = grads2img(lin2img(pre_grad, image_resolution=resolution)).permute(1, 2, 0).numpy()

            # predict_init_depth_copy = predict_init_depth_copy.squeeze(1).detach().cpu()
            # pre_sobel_gradx = scipy.ndimage.sobel(predict_init_depth_copy.numpy(), axis=1).squeeze(0)[..., None]
            # pre_sobel_grady = scipy.ndimage.sobel(predict_init_depth_copy.numpy(), axis=2).squeeze(0)[..., None]
            # pre_sobelgt_gradient = torch.cat((torch.from_numpy(pre_sobel_gradx).reshape(-1, 1),
            #                                     torch.from_numpy(pre_sobel_grady).reshape(-1, 1)),
            #                                     dim=-1).unsqueeze(0)
            # pre_sobel_img = grads2img(lin2img(pre_sobelgt_gradient, image_resolution=resolution)).permute(1, 2, 0).numpy()

            # gt_laplace_img = cv2.cvtColor(cv2.applyColorMap(to_uint8(rescale_img(
            # lin2img(gt_laplace, image_resolution=resolution), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), \
            # cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

            # pre_laplace_img = cv2.cvtColor(cv2.applyColorMap(to_uint8(rescale_img(
            # lin2img(pre_laplace, image_resolution=resolution), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), \
            # cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

            depth_sample_whole = depth_sample_whole.squeeze().detach().cpu().numpy()
            depth_sample_whole = cv2.cvtColor(depth_sample_whole, cv2.COLOR_GRAY2BGR)

            if (i % 20 == 0):
            # if (epoch % 10 == 0):
                writer.add_image('groundtruth and out', 
                    # np.concatenate((depth_scale2, depth_sample_whole, predict_depth, pre_scale4, pre_scale8), axis=1),
                    np.concatenate((depth, predict_init_depth, finetune_depth), axis=1), #, conv_init_depth
                    global_step=i+train_pics*epoch,
                    dataformats='HWC')

                writer.add_image('gt_grad and pre_grad',
                    # np.concatenate((gt_visual_init_sobel_img, pre_visual_init_sobel_img, pre_gradient_img, depth_sample_whole), axis=1),
                    np.concatenate((gt_visual_init_sobel_img, pre_visual_init_sobel_img, depth_sample_whole), axis=1),
                    global_step=i+train_pics*epoch,
                    dataformats='HWC')

        for p in net_optimizer.param_groups:
            p['lr'] *= 0.9
        for p in finetune_optimizer.param_groups:
            p['lr'] *= 0.9
        # for p in convinit_optimizer.param_groups:
        #     p['lr'] *= 0.9
        
        # evaluation
        net.eval()
        finetune.eval()
        # convinit.eval()

        nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                    transform=transforms.Compose([nyu.Rescale_update(train=False), 
                                                                nyu.ToTensor_update(),
                                                                nyu.SamplePoint_update(sample_num)]))
        dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=1, shuffle=False, drop_last=True)

        it = iter(dataloader_eval)

        pbar = tqdm(total=eval_pics)

        rmse_total = 0.0
        mae_total = 0.0
        irmse_total = 0.0
        imae_total = 0.0
        rel_total = 0.0
        del_1_total = 0.0
        del_2_total = 0.0
        del_3_total = 0.0

        rmse_total_finetune = 0.0
        mae_total_finetune = 0.0
        irmse_total_finetune = 0.0
        imae_total_finetune = 0.0
        rel_total_finetune = 0.0
        del_1_total_finetune = 0.0
        del_2_total_finetune = 0.0
        del_3_total_finetune = 0.0

        for i in range(eval_pics):
            with torch.no_grad():
                try:
                    sample_dict = next(it)
                except:
                    continue

                depth = sample_dict['depth'].to(device)
                rgb = sample_dict['rgb'].to(device)
                index_list = sample_dict['index_list'].to(device)
                depth_sample_whole = sample_dict['depth_sample_whole'].to(device)
                gt_gradient = sample_dict['gt_gradient'].to(device)

                index_list = index_list / torch.tensor([resolution[0]-1, resolution[1]-1]).reshape(1, 2, 1, 1).to(device)
                # index_list = index_list * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)

                index_list_copy = index_list.clone()

                loss1, predict_init_depth = net(rgb, depth_sample_whole, index_list, index_list_copy)
                # conv_init_depth = convinit(predict_init_depth)
                finetune_depth = finetune(predict_init_depth, rgb)

                loss_dict = evaluation(depth, predict_init_depth)
                loss_dict_finetune = evaluation(depth, finetune_depth)

                rmse = loss_dict['rmse'].item()
                mae = loss_dict["mae"].item()
                irmse = loss_dict["irmse"].item()
                imae = loss_dict["imae"].item()
                rel = loss_dict["rel"].item()
                del_1 = loss_dict["del_1"].item()
                del_2 = loss_dict["del_2"].item()
                del_3 = loss_dict["del_3"].item()

                rmse_total += rmse
                mae_total += mae
                irmse_total += irmse
                imae_total += imae
                rel_total += rel
                del_1_total += del_1
                del_2_total += del_2
                del_3_total += del_3

                rmse_finetune = loss_dict_finetune['rmse'].item()
                mae_finetune = loss_dict_finetune["mae"].item()
                irmse_finetune = loss_dict_finetune["irmse"].item()
                imae_finetune = loss_dict_finetune["imae"].item()
                rel_finetune = loss_dict_finetune["rel"].item()
                del_1_finetune = loss_dict_finetune["del_1"].item()
                del_2_finetune = loss_dict_finetune["del_2"].item()
                del_3_finetune = loss_dict_finetune["del_3"].item()

                rmse_total_finetune += rmse_finetune
                mae_total_finetune += mae_finetune
                irmse_total_finetune += irmse_finetune
                imae_total_finetune += imae_finetune
                rel_total_finetune += rel_finetune
                del_1_total_finetune += del_1_finetune
                del_2_total_finetune += del_2_finetune
                del_3_total_finetune += del_3_finetune

                error_str = 'RMSE={:.3f} | REL={:.3f} | D1={:.3f} | D2={:.3f} | D3={:.3f}'.format(
                            rmse, rel, del_1, del_2, del_3)
                pbar.set_description(error_str)
                pbar.update(1)

        rmse_total = rmse_total / eval_pics
        rel_total = rel_total / eval_pics
        del_1_total = del_1_total / eval_pics
        del_2_total = del_2_total / eval_pics
        del_3_total = del_3_total / eval_pics

        rmse_total_finetune = rmse_total_finetune / eval_pics
        rel_total_finetune = rel_total_finetune / eval_pics
        del_1_total_finetune = del_1_total_finetune / eval_pics
        del_2_total_finetune = del_2_total_finetune / eval_pics
        del_3_total_finetune = del_3_total_finetune / eval_pics

        print("rmse is: ", rmse_total)
        print("rel is: ", rel_total)
        print("d1 is: ", del_1_total)
        print("d2 is: ", del_2_total)
        print("d3 is: ", del_3_total)

        writer.add_scalar('RMSE', rmse_total, global_step=epoch)
        writer.add_scalar('REL', rel_total, global_step=epoch)
        writer.add_scalar('DELTA1', del_1_total, global_step=epoch)
        writer.add_scalar('DELTA2', del_2_total, global_step=epoch)
        writer.add_scalar('DELTA3', del_3_total, global_step=epoch)

        writer.add_scalar('RMSE_finetune', rmse_total_finetune, global_step=epoch)
        writer.add_scalar('REL_finetune', rel_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA1_finetune', del_1_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA2_finetune', del_2_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA3_finetune', del_3_total_finetune, global_step=epoch)

        if (rmse_total < 0.2 and rel_total < 0.06):
            savefilename = './checkpoints/' + run_file_name + '_net' + '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                            + "_epoch_" + str(epoch)+'.tar'
            torch.save(net.state_dict(), savefilename)

            savefilename = './checkpoints/' + run_file_name + '_finetune' '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                            + "_epoch_" + str(epoch)+'.tar'
            torch.save(finetune.state_dict(), savefilename)

            # savefilename = './checkpoints/' + run_file_name + '_convinit' '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
            #                 + "_epoch_" + str(epoch)+'.tar'
            # torch.save(convinit.state_dict(), savefilename)


def May_Net():
    feature_ex_net = fc.FeatureExtractionNet().to(device)
    # fc_net = fc.FullConnected_baseline().to(device)
    fc_net = fc.Conv_1d().to(device)
    finetune_net = fc.Finetune().to(device)

    feature_ex_net.load_state_dict(torch.load("./checkpoints/pretrained/feature_extract_pretrained.tar"))
    # fc_net.load_state_dict(torch.load("./checkpoints/pretrained/fullconnect_pretrained.tar"))
    finetune_net.load_state_dict(torch.load("./checkpoints/pretrained/finetune_pretrained.tar"))

    feature_ex_net_optimizer = torch.optim.Adam(feature_ex_net.parameters(), lr=1e-5, weight_decay=5e-4)
    fc_net_optimizer = torch.optim.Adam(fc_net.parameters(), lr=4e-5, weight_decay=5e-4)
    finetune_net_optimizer = torch.optim.Adam(finetune_net.parameters(), lr=1e-5, weight_decay=5e-4)

    nyu_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_train.txt",
                                    transform=transforms.Compose([nyu.Rescale_update(train=True),
                                                                nyu.ToTensor_update(),
                                                                nyu.SamplePoint_update(sample_num)]))
    dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)
    
    lossfunction = LossFunction().to(device)
    writer = SummaryWriter(log_dir='runs/' + run_file_name)

    sobel = Sobel().to(device)

    for epoch in range(total_epochs):
        feature_ex_net.train()
        fc_net.train()
        finetune_net.train()
        it = iter(dataloader)

        for i in range(train_pics):
            time_1 = time.time()
            try:
                sample_dict = next(it)
            except:
                continue

            depth = sample_dict['depth'].to(device)
            rgb = sample_dict['rgb'].to(device)
            index_list = sample_dict['index_list'].to(device)
            depth_sample_whole = sample_dict['depth_sample_whole'].to(device)
            gt_gradient = sample_dict['gt_gradient'].to(device)

            filter_mask = depth > 0
            if (filter_mask.sum() < 67000):
                print("Bad Data!!!!!!!!!")
                continue

            index_list = index_list / torch.tensor([resolution[0]-1, resolution[1]-1]).reshape(1, 2, 1, 1).to(device)
            # index_list = index_list * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)

            time_2 = time.time()

            ###############################
            # get feature map from rgb
            featuremap = feature_ex_net(rgb)

            sample_mask = depth_sample_whole >= 0
            depth_sample = torch.masked_select(depth_sample_whole, sample_mask)
            depth_sample = depth_sample.reshape(sample_num, 1)
            feature_map_with_pose = torch.cat((featuremap, position_encoding(index_list)), dim=1)

            feature_map_with_pose_layers = feature_map_with_pose.size()[1] # 256 + position_layers
            fc_in = feature_map_with_pose.reshape(feature_map_with_pose_layers, 228*304).permute(1, 0)
            fc_in = fc_in.permute(1, 0).unsqueeze(0)
            fc_out = fc_net(fc_in) # 69312 * 256
            fc_out = fc_out.squeeze().permute(1, 0)
            # print("fffffffffffffffffff", fc_out.size())

            fc_out_sample = fc_out.permute(1, 0).view(1, 256, 228, 304)
            fc_out_sample = torch.masked_select(fc_out_sample, sample_mask)
            fc_out_sample = fc_out_sample.reshape(fc_out.size()[1], sample_num).permute(1, 0) # 1000 * 256

            # least_squres to GET W!
            fc_out_sample_trans = fc_out_sample.permute(1, 0)
            new_a = torch.matmul(fc_out_sample_trans, fc_out_sample)
            new_b = torch.matmul(fc_out_sample_trans, depth_sample)
            w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(new_a.size()[0]).to(device)), new_b)

            predict_sample = torch.matmul(fc_out_sample, w)
            loss_lq = lossfunction(depth_sample, predict_sample)

            # predict depth
            predict_depth = torch.matmul(fc_out, w).reshape(1, 1, 228, 304)
            finetune_depth = finetune_net(predict_depth, rgb)

            # caculate loss
            mask = depth > 0
            loss1 = lossfunction(depth, predict_depth, mask=mask)
            loss2 = lossfunction(depth, finetune_depth, mask=mask)

            loss = loss1 + loss2 + loss_lq

            feature_ex_net.zero_grad()
            fc_net.zero_grad()
            finetune_net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_ex_net.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(fc_net.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(finetune_net.parameters(), 10)
            feature_ex_net_optimizer.step()
            fc_net_optimizer.step()
            finetune_net_optimizer.step()

            time_4 = time.time()

            print(run_file_name + ' Epoch:'+str(epoch)+' Pic:'+str(i)+" Using: %.2f s"%(time.time()-time_1))
            print("---------loss   : %.4f" % loss.item())

            writer.add_scalar('loss', loss.item(), global_step=i+train_pics*epoch)

            # visualization
            depth = depth / 10
            depth = depth.squeeze().detach().cpu().numpy()
            depth = (255.0*cm(depth)).astype('uint8')

            predict_depth = predict_depth / 10
            predict_depth = predict_depth.squeeze().detach().cpu().numpy()
            predict_depth = (255.0*cm(predict_depth)).astype('uint8')

            finetune_depth = finetune_depth / 10
            finetune_depth = finetune_depth.squeeze().detach().cpu().numpy()
            finetune_depth = (255.0*cm(finetune_depth)).astype('uint8')
            ################################################################

            depth_sample_whole = depth_sample_whole.squeeze().detach().cpu().numpy()
            depth_sample_whole = cv2.cvtColor(depth_sample_whole, cv2.COLOR_GRAY2BGR)

            if (i % 20 == 0):
            # if (epoch % 10 == 0):
                writer.add_image('groundtruth and out', 
                    np.concatenate((depth, predict_depth, finetune_depth), axis=1),
                    global_step=i+train_pics*epoch,
                    dataformats='HWC')

                writer.add_image('gt_grad and pre_grad',
                    depth_sample_whole,
                    global_step=i+train_pics*epoch,
                    dataformats='HWC')

        # for p in feature_ex_net_optimizer.param_groups:
        #     p['lr'] *= 0.9
        # for p in fc_net_optimizer.param_groups:
        #     p['lr'] *= 0.9
        # for p in finetune_net_optimizer.param_groups:
        #     p['lr'] *= 0.9
        
        # evaluation
        feature_ex_net.eval()
        fc_net.eval()
        finetune_net.eval()

        nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                    transform=transforms.Compose([nyu.Rescale_update(train=False), 
                                                                nyu.ToTensor_update(),
                                                                nyu.SamplePoint_update(sample_num)]))
        dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=1, shuffle=False, drop_last=True)

        it = iter(dataloader_eval)

        pbar = tqdm(total=eval_pics)

        rmse_total = 0.0
        mae_total = 0.0
        irmse_total = 0.0
        imae_total = 0.0
        rel_total = 0.0
        del_1_total = 0.0
        del_2_total = 0.0
        del_3_total = 0.0

        rmse_total_finetune = 0.0
        mae_total_finetune = 0.0
        irmse_total_finetune = 0.0
        imae_total_finetune = 0.0
        rel_total_finetune = 0.0
        del_1_total_finetune = 0.0
        del_2_total_finetune = 0.0
        del_3_total_finetune = 0.0

        for i in range(eval_pics):
            with torch.no_grad():
                try:
                    sample_dict = next(it)
                except:
                    continue

                depth = sample_dict['depth'].to(device)
                rgb = sample_dict['rgb'].to(device)
                index_list = sample_dict['index_list'].to(device)
                depth_sample_whole = sample_dict['depth_sample_whole'].to(device)
                gt_gradient = sample_dict['gt_gradient'].to(device)

                index_list = index_list / torch.tensor([resolution[0]-1, resolution[1]-1]).reshape(1, 2, 1, 1).to(device)
                # index_list = index_list * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)

                featuremap = feature_ex_net(rgb)

                sample_mask = depth_sample_whole >= 0
                depth_sample = torch.masked_select(depth_sample_whole, sample_mask)
                depth_sample = depth_sample.reshape(sample_num, 1)
                feature_map_with_pose = torch.cat((featuremap, position_encoding(index_list)), dim=1)

                feature_map_with_pose_layers = feature_map_with_pose.size()[1]
                fc_in = feature_map_with_pose.reshape(feature_map_with_pose_layers, 228*304).permute(1, 0)
                fc_in = fc_in.permute(1, 0).unsqueeze(0)
                fc_out = fc_net(fc_in)
                fc_out = fc_out.squeeze().permute(1, 0)

                fc_out_sample = fc_out.permute(1, 0).view(1, 256, 228, 304)
                fc_out_sample = torch.masked_select(fc_out_sample, sample_mask)
                fc_out_sample = fc_out_sample.reshape(fc_out.size()[1], sample_num).permute(1, 0) # 1000 * 256

                # least_squres to GET W!
                fc_out_sample_trans = fc_out_sample.permute(1, 0)
                new_a = torch.matmul(fc_out_sample_trans, fc_out_sample)
                new_b = torch.matmul(fc_out_sample_trans, depth_sample)
                w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(new_a.size()[0]).to(device)), new_b)

                # predict depth
                predict_depth = torch.matmul(fc_out, w).reshape(1, 1, 228, 304)
                finetune_depth = finetune_net(predict_depth, rgb)

                loss_dict = evaluation(depth, predict_depth)
                loss_dict_finetune = evaluation(depth, finetune_depth)

                rmse = loss_dict['rmse'].item()
                mae = loss_dict["mae"].item()
                irmse = loss_dict["irmse"].item()
                imae = loss_dict["imae"].item()
                rel = loss_dict["rel"].item()
                del_1 = loss_dict["del_1"].item()
                del_2 = loss_dict["del_2"].item()
                del_3 = loss_dict["del_3"].item()

                rmse_total += rmse
                mae_total += mae
                irmse_total += irmse
                imae_total += imae
                rel_total += rel
                del_1_total += del_1
                del_2_total += del_2
                del_3_total += del_3

                rmse_finetune = loss_dict_finetune['rmse'].item()
                mae_finetune = loss_dict_finetune["mae"].item()
                irmse_finetune = loss_dict_finetune["irmse"].item()
                imae_finetune = loss_dict_finetune["imae"].item()
                rel_finetune = loss_dict_finetune["rel"].item()
                del_1_finetune = loss_dict_finetune["del_1"].item()
                del_2_finetune = loss_dict_finetune["del_2"].item()
                del_3_finetune = loss_dict_finetune["del_3"].item()

                rmse_total_finetune += rmse_finetune
                mae_total_finetune += mae_finetune
                irmse_total_finetune += irmse_finetune
                imae_total_finetune += imae_finetune
                rel_total_finetune += rel_finetune
                del_1_total_finetune += del_1_finetune
                del_2_total_finetune += del_2_finetune
                del_3_total_finetune += del_3_finetune

                error_str = 'RMSE={:.3f} | REL={:.3f} | D1={:.3f} | D2={:.3f} | D3={:.3f}'.format(
                            rmse, rel, del_1, del_2, del_3)
                pbar.set_description(error_str)
                pbar.update(1)

        rmse_total = rmse_total / eval_pics
        rel_total = rel_total / eval_pics
        del_1_total = del_1_total / eval_pics
        del_2_total = del_2_total / eval_pics
        del_3_total = del_3_total / eval_pics

        rmse_total_finetune = rmse_total_finetune / eval_pics
        rel_total_finetune = rel_total_finetune / eval_pics
        del_1_total_finetune = del_1_total_finetune / eval_pics
        del_2_total_finetune = del_2_total_finetune / eval_pics
        del_3_total_finetune = del_3_total_finetune / eval_pics

        print("rmse is: ", rmse_total)
        print("rel is: ", rel_total)
        print("d1 is: ", del_1_total)
        print("d2 is: ", del_2_total)
        print("d3 is: ", del_3_total)

        writer.add_scalar('RMSE', rmse_total, global_step=epoch)
        writer.add_scalar('REL', rel_total, global_step=epoch)
        writer.add_scalar('DELTA1', del_1_total, global_step=epoch)
        writer.add_scalar('DELTA2', del_2_total, global_step=epoch)
        writer.add_scalar('DELTA3', del_3_total, global_step=epoch)

        writer.add_scalar('RMSE_finetune', rmse_total_finetune, global_step=epoch)
        writer.add_scalar('REL_finetune', rel_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA1_finetune', del_1_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA2_finetune', del_2_total_finetune, global_step=epoch)
        writer.add_scalar('DELTA3_finetune', del_3_total_finetune, global_step=epoch)

        if (rmse_total < 0.2 and rel_total < 0.06):
        # if (True):
            savefilename = './checkpoints/' + run_file_name + '_feature' + '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                            + "_epoch_" + str(epoch)+'.tar'
            torch.save(feature_ex_net.state_dict(), savefilename)

            savefilename = './checkpoints/' + run_file_name + '_fc' '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                            + "_epoch_" + str(epoch)+'.tar'
            torch.save(fc_net.state_dict(), savefilename)

            savefilename = './checkpoints/' + run_file_name + '_finetune' '_rmse_' + str(rmse_total) + '_rel_' + str(rel_total) \
                            + "_epoch_" + str(epoch)+'.tar'
            torch.save(finetune_net.state_dict(), savefilename)

if __name__ == '__main__':
    # train_main()
    # train_multiscale()
    # train_update()
    # feature_ex_net = fc.FeatureExtractionNet().to(device)
    # fullconnect_baseline = fc.FullConnected_baseline().to(device)
    # fc_dict = fullconnect_baseline.state_dict()
    # pretrained_dict = torch.load("./checkpoints/shuabang/baseline_net_rmse_0.19360252093261718_rel_0.05172041724991361_epoch_17.tar")
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in fc_dict}
    # fc_dict.update(pretrained_dict)
    # fullconnect_baseline.load_state_dict(fc_dict)
    # torch.save(fullconnect_baseline.state_dict(), "./checkpoints/pretrained/fullconnect_pretrained.tar")
    # feature_dict = feature_ex_net.state_dict()
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in feature_dict}
    # feature_dict.update(pretrained_dict)
    # feature_ex_net.load_state_dict(feature_dict)
    # torch.save(feature_ex_net.state_dict(), "feature_extract_pretrained.tar")
    May_Net()