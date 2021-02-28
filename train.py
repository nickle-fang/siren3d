from numpy.lib.function_base import disp
import torch
import random
import time
import cv2
import math
import numpy as np
from torch._C import dtype
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
sample_num = 300
final_layer = 256 # also in other files
feature_map_layers = 256
resolution = 224
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
    resnet = fc.Resnet(resolution)
    resnet = resnet.to(device)
    resnet.eval()

    base_func = fc.FC_basefunction(final_layer, feature_map_layers+40)

    # base_func.load_state_dict(torch.load("./checkpoints/256_w_0.tar"))

    base_func = base_func.to(device)
    base_func.train()

    base_func_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.00001, weight_decay=5e-4)

    # writer1 = SummaryWriter()

    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.L1Loss()

    for epoch in range(total_epochs):
        # flying_dataset = fly.FlyingDataset(filelistpath="./data/flying3d_train.txt",
        #                                     transform=transforms.Compose([
        #                                         fly.Rescale((resolution, resolution))
        #                                     ]))
        # dataloader = DataLoader(flying_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
        # it = iter(dataloader)

        nyu_dataset = nyu.NYUDataset(transform=transforms.Compose([nyu.Rescale((resolution,resolution)), nyu.ToTensor()]))
        dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=shuffle_bool, drop_last=True)
        it = iter(dataloader)

        for i in range(total_pics):
            time_start = time.time()
            # calculate the W using known depth
            point_sample_index = random.sample([[i, j] for i in range(resolution) for j in range(resolution)], sample_num)

            sample_dict = next(it)
            # left_img = sample_dict['left_img']
            left_img = sample_dict['rgb_img']
            disp_gt = sample_dict['disp_gt']
            left_img = left_img.to(device, torch.float32)
            disp_gt = disp_gt.to(device, torch.float32)

            time_1 = time.time()

            feature_map = resnet(left_img).cpu() # 1*256*resolution*resolution

            time_2 = time.time()

            # point_sample_index_norm ： 1*2*sample_num
            point_sample_index_norm = torch.from_numpy(np.array(point_sample_index).transpose(1, 0)/float(resolution)).unsqueeze(0)
            point_sample_index_norm = point_sample_index_norm.to(device, dtype=torch.float32)

            fc_in = torch.zeros(1, feature_map_layers, sample_num)
            depth_sample = torch.zeros(sample_num, 1)
            fc_i = 0
            for p in point_sample_index:
                fc_in[:, :, fc_i] = feature_map[:, :, p[0], p[1]]
                depth_sample[fc_i, :] = disp_gt[:, :, p[0], p[1]]
                fc_i += 1

            fc_in, depth_sample = fc_in.to(device), depth_sample.to(device)
            fc_in = torch.cat((fc_in, position_encoding(point_sample_index_norm)), dim=1).squeeze()
            fc_in = torch.transpose(fc_in, 1, 0).to(device)

            time_3 = time.time()

            ##################### add position encoding

            fc_concat = base_func(fc_in) # sample_num * final_layers
            fc_concat = torch.sin(fc_concat)

            # fc_concat_trans = fc_concat.permute(1, 0) # final_layers * sample_num
            # new_a = torch.matmul(fc_concat_trans, fc_concat)
            # new_b = torch.matmul(fc_concat_trans, depth_sample)
            # w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(final_layer).to(device)), new_b) # final_layers * 1

            w, _, __, ___ = np.linalg.lstsq(fc_concat.cpu().detach().numpy(), depth_sample.cpu().detach().numpy())
            w = torch.from_numpy(w).to(device)

            # w, _ = torch.lstsq(depth_sample, fc_concat)
            # w = w[:fc_concat.size()[1], :]
            predict_sample_depth = torch.matmul(fc_concat, w)

            # TODO delete the grad of w !!!
            w = w.detach()

            time_4 = time.time()

            loss_1 = loss_function(predict_sample_depth, depth_sample)

            # solving_loss = torch.sum(torch.abs(torch.matmul(fc_concat, w) - depth_sample)) / sample_num

            time_5 = time.time()

            # predict depth using W
            feature_map = feature_map.to(device)
            feature_map = feature_map.reshape(1, feature_map_layers, resolution*resolution)
            index_list = np.array([[i / resolution, j / resolution] for i in range(resolution) for j in range(resolution)]).transpose(1, 0)
            index_list = torch.from_numpy(index_list).unsqueeze(0).to(device, torch.float32)
            
            fc_pic_in = torch.cat((feature_map, position_encoding(index_list)), axis=1).squeeze()
            fc_pic_in = torch.transpose(fc_pic_in, 1, 0)

            ##################### add position encoding

            fc_pic_out = base_func(fc_pic_in)
            fc_pic_out = torch.sin(fc_pic_out)
            predict_depth = torch.matmul(fc_pic_out, w).reshape(1, resolution, resolution).unsqueeze(0)

            loss = loss_function(predict_depth, disp_gt)

            time_6 = time.time()

            #######################################
            # fc_i = 0
            # predict_depth_on_sample_index = torch.zeros(1, sample_num)
            # for p in point_sample_index:
            #     predict_depth_on_sample_index[:, fc_i] = predict_depth[:, :, p[0], p[1]]
            #     fc_i += 1
            # loss_temp = loss_function(predict_depth_on_sample_index, depth_sample_copy)
            # writer1.add_scalar('loss_temp', loss_temp.item(), global_step=i+total_pics*epoch)
            #######################################

            # writer1.add_scalar('loss', loss.item(), global_step=i+total_pics*epoch)

            # disp_gt = disp_gt.squeeze().cpu().detach().numpy()
            # predict_depth = predict_depth.squeeze().cpu().detach().numpy()
            # disp_color = cv2.applyColorMap(disp_gt.astype(np.uint8), 4)
            # predict_color = cv2.applyColorMap(predict_depth.astype(np.uint8), 4)

            # disp_gt = cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # predict_depth = cv2.normalize(predict_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # if (i % 10 == 0):
            #     writer1.add_image('groundtruth and out', 
            #                         np.concatenate((disp_gt, predict_depth), axis=1, out=None),
            #                         global_step=i+total_pics*epoch,
            #                         dataformats='HW')

            base_func_optimizer.zero_grad()
            loss_1.backward()
            torch.nn.utils.clip_grad_norm_(base_func.parameters(), 10)
            base_func_optimizer.step()

            print('Epoch:'+str(epoch)+' Pic:'+str(i)+" Processing a single picture using: %.2f s"%(time.time()-time_start))
            print("---------Single picture loss: %.4f" % loss.item())
            # print("---------solvingloss        : %.4f" % solving_loss)
            print("---------loss_1             : %.4f" % loss_1.item())

            print("0-1: %.4f " % (time_1 - time_start))
            print("1-2: %.4f " % (time_2 - time_1))
            print("2-3: %.4f " % (time_3 - time_2)) #######
            print("3-4: %.4f " % (time_4 - time_3))
            print("4-5: %.4f " % (time_5 - time_4))
            print("5-6: %.4f " % (time_6 - time_5)) ########
            print("6-7: %.4f " % (time.time() - time_6)) ########
        
        # if (epoch % 10 == 0):
        #     savefilename = './checkpoints/new_256_'+str(epoch)+'.tar'
        #     torch.save(base_func.state_dict(), savefilename)
            # cv2.imwrite(str(epoch)+"_single.png", cv2.normalize(predict_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            # cv2.imwrite(str(epoch)+"_ground_truth_2000.png", cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            

if __name__ == '__main__':
    train_main()
