#!/home/nickle/miniconda3/envs/siren3d/bin/python

import torch
import time
import os
import cv2
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.transforms.transforms import RandomCrop

from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
shuffle_bool = False
eval_bool = True
train_pics = 47584
eval_pics = 654
total_epochs = 20000
sample_num = 200
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (120, 160)
raw_resolution = (480, 640)
reso_scale2 = (240, 320)
reso_scale4 = (120, 160)
reso_scale8 = (64, 80)
lamda1 = 1 # loss_1
lamda2 = 1 # loss_2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_file_name = "multiscale_160"
data_gain_bool = True

# define loss function
# loss_function = torch.nn.SmoothL1Loss(beta=0.001)
# loss_function = torch.nn.L1Loss()
# l2_loss_function = torch.nn.MSELoss()

class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss_function = torch.nn.L1Loss()
    
    def forward(self, gt, predict):
        mask = gt > 0
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
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
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
    for i in range(4):
        encoded = torch.cat((encoded, 
                            torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
                            torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

        # encoded = torch.cat((encoded, x, x, y, y), dim=1)

    return encoded.permute(0, 2, 1)


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
    learning_rate = 3e-3
    net = fc.PyramidDepthEstimation(reso_scale2, reso_scale4, reso_scale8, raw_resolution, sample_num)
    net = net.to(device)
    # net.load_state_dict(torch.load("./checkpoints/finetune_resnet_gradloss_4.tar"))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

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

        if (epoch > 3):
            learning_rate = 4e-4
        if (epoch > 5):
            learning_rate = 4e-5
        if (epoch > 8):
            learning_rate = 4e-6
        net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

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
            loss1, depth_finetune_1, depth_finetune_2, predict_init_depth = net(rgb_scale4, rgb_scale2, depth_sample_whole)

            predict_depth_kid = upsample_visual(depth_finetune_1)
            predict_depth_final = upsample_visual(depth_finetune_2)
            predict_init_depth = upsample_visual(predict_init_depth)

            time_3 = time.time()

            loss2 = lossfunction(disp_gt_raw, predict_init_depth)
            loss3 = lossfunction(disp_gt_raw, predict_depth_kid)
            loss4 = lossfunction(disp_gt_raw, predict_depth_final)

            # loss3 = lossfunction(depth_scale4, pre_scale4)
            # loss4 = lossfunction(depth_scale8, pre_scale8)

            # gradloss2 = gradloss(depth_scale2, predict_init_depth)
            # gradloss2_final = gradloss(depth_scale2, predict_depth)

            # gradloss3 = gradloss(depth_scale4, pre_scale4)
            # gradloss4 = gradloss(depth_scale8, pre_scale8)

            # loss = loss1 + loss4 + loss3 + loss2 + gradloss2 + gradloss3 + gradloss4
            loss = loss1 + loss2 #+ loss3 + gradloss2 + gradloss2_final # + loss3

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            net_optimizer.step()

            print(run_file_name + ' Epoch:'+str(epoch)+' Pic:'+str(i)+" Using: %.2f s"%(time.time()-time_1))
            print("---------loss             : %.4f" % loss.item())

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

            rgb_scale2 = rgb_scale2.squeeze().permute(1, 2, 0).detach().cpu().numpy()

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

        savefilename = './checkpoints/' + run_file_name + "_" + str(epoch)+'.tar'
        torch.save(net.state_dict(), savefilename)

        # evaluation
        # nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
        #                             transform=transforms.Compose([nyu.Rescale_Multiscale \
        #                                                          (raw_resolution, reso_scale2, reso_scale4, reso_scale8), 
        #                                                     nyu.ToTensor_Multiscale(),
        #                                                     nyu.SamplePoint_Multiscale \
        #                                                         (reso_scale8, sample_num)]))
        # dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

        nyu_dataset_eval = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                             transform=transforms.Compose([nyu.Rescale(reso_scale2, 
                                                                                       data_gain=False), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=reso_scale2, 
                                                                                    sample_=sample_num)]))
        dataloader_eval = DataLoader(nyu_dataset_eval, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

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
                _, depth_finetune_1, depth_finetune_2, predict_init_depth = net(rgb_scale4, rgb_scale2, depth_sample_whole)
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


if __name__ == '__main__':
    # train_main()
    train_multiscale()
