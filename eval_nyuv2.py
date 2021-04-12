import torch
import time
import os
import math
import numpy as np
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
shuffle_bool = False
eval_bool = True
eval_pics = 654
sample_num = 1000
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = (480, 640)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def evaluation(gt, predict_depth, t_valid):
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


def eval_main_loop(dataloader, feature_ex_net, base_func):
    it = iter(dataloader)
    total_rmse = 0.0
    total_rel = 0.0
    total_d1 = 0.0
    total_d2 = 0.0
    total_d3 = 0.0
    time_start = time.time()

    pbar = tqdm(total=eval_pics)

    for i in range(eval_pics):
        with torch.no_grad():
            sample_dict = next(it)

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

            sample_mask = depth_sample_whole > 0
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

            # predict depth using W
            feature_map = feature_map.reshape(1, feature_map_layers, resolution[0]*resolution[1]).permute(0, 2, 1)
            index_list = index_list / torch.tensor([[resolution[0]], [resolution[1]]]).to(device, dtype=torch.float32)
            fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=2).squeeze()
            fc_pic_out = base_func(fc_pic_in)

            # add pose again
            fc_pic_out = torch.cat((fc_pic_out, position_encoding(index_list).squeeze()), dim=1)

            predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 1, resolution[0], resolution[1])

            loss_dict = evaluation(disp_gt, predict_depth, 0.0001)
            
            rmse = loss_dict['rmse']
            mae = loss_dict["mae"]
            irmse = loss_dict["irmse"]
            imae = loss_dict["imae"]
            rel = loss_dict["rel"]
            del_1 = loss_dict["del_1"]
            del_2 = loss_dict["del_2"]
            del_3 = loss_dict["del_3"]

            total_rmse += rmse.item()
            total_rel += rel.item()
            total_d1 += del_1.item()
            total_d2 += del_2.item()
            total_d3 += del_3.item()

            error_str = 'RMSE= {:.3f} | REL= {:.3f} | D1= {:.3f} | D2= {:.3f} | D3= {:.3f}'.format(
                        rmse, rel, del_1, del_2, del_3)
            pbar.set_description(error_str)
            pbar.update(1)

    print("RMSE is: ", total_rmse / eval_pics)
    print("REL is: ", total_rel / eval_pics)
    print("D1 is: ", total_d1 / eval_pics)
    print("D2 is: ", total_d2 / eval_pics)
    print("D3 is: ", total_d3 / eval_pics)
    print("Using time: ", time.time() - time_start)


def eval_main():
    base_func = fc.FC_basefunction(final_layer, feature_map_layers+20)
    base_func.load_state_dict(torch.load("./checkpoints/base_from_base_gain_rmse_0.16423152047935702_rel_0.039291640183965276_2.tar"))
    base_func = base_func.to(device)

    feature_ex_net = fc.DRPNet()
    feature_ex_net.load_state_dict(torch.load("./checkpoints/unet_from_base_gain_rmse_0.16423152047935702_rel_0.039291640183965276_2.tar"))
    feature_ex_net = feature_ex_net.to(device)

    nyu_eval_dataset = nyu.NYUDatasetAll(filelistpath="./data/nyudepthv2_valid.txt",
                                      transform=transforms.Compose([nyu.Rescale(resolution, 
                                                                                size_scale2=(100, 100),
                                                                                size_scale4=(100, 100),
                                                                                data_gain=False), 
                                                                    nyu.ToTensor(),
                                                                    nyu.SamplePoint(reso=resolution, 
                                                                                    sample_=sample_num)]))
    eval_dataloader = DataLoader(nyu_eval_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)

    base_func.eval()
    feature_ex_net.eval()
    eval_main_loop(eval_dataloader, feature_ex_net, base_func)

if __name__ == '__main__':
    eval_main()
