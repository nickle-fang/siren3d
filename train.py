import torch
import random
import time
import cv2
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataloader import flying3d_dataloader as fly
from dataloader import nyu_dataloader as nyu
from model import fc_basefunction as fc


batch_num = 1
total_pics = 1448
total_epochs = 300
sample_num = 1000
final_layer = 64 # also in other files
feature_map_layers = 67
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_main():
    resnet = fc.Resnet()
    resnet = resnet.to(device)
    resnet.eval()

    base_func = fc.FC_basefunction(final_layer)
    base_func = base_func.to(device)
    base_func.train()

    base_func_optimizer = torch.optim.Adam(base_func.parameters(), lr=0.0001, weight_decay=5e-4)

    writer1 = SummaryWriter()

    loss_function = torch.nn.MSELoss()

    for epoch in range(total_epochs):
        # flying_dataset = fly.FlyingDataset(filelistpath="./data/flying3d_train.txt",
        #                                     transform=transforms.Compose([
        #                                         fly.Rescale((224, 224))
        #                                     ]))
        # dataloader = DataLoader(flying_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
        # it = iter(dataloader)

        nyu_dataset = nyu.NYUDataset(transform=transforms.Compose([nyu.Rescale((224,224)), nyu.ToTensor()]))
        dataloader = DataLoader(nyu_dataset, batch_size=batch_num, shuffle=True, drop_last=True)
        it = iter(dataloader)

        for i in range(total_pics):
            time_start = time.time()
            # processing a single picture
            # calculate the W using known depth
            point_sample_index = random.sample([[i, j] for i in range(224) for j in range(224)], sample_num)

            sample_dict = next(it)
            # left_img = sample_dict['left_img']
            left_img = sample_dict['rgb_img']
            disp_gt = sample_dict['disp_gt']
            
            left_img = left_img.to(device, torch.float32)
            disp_gt = disp_gt.to(device, torch.float32)

            feature_map = resnet(left_img).cpu()

            fc_concat = torch.zeros(1, sample_num, final_layer)
            depth_sample = torch.zeros(1, sample_num)
            fc_i = 0
            for p in point_sample_index:
                fc_in = torch.zeros(1, feature_map_layers+2)
                fc_in[:, :feature_map_layers] = feature_map[:, :, p[0], p[1]]
                fc_in[:, feature_map_layers] = p[0]
                fc_in[:, feature_map_layers+1] = p[1]

                fc_in = fc_in.to(device)
                fc_out = base_func(fc_in)

                fc_concat[:, fc_i, :] = fc_out
                depth_sample[:, fc_i] = disp_gt[:, :, p[0], p[1]]

                fc_i += 1

            fc_concat = fc_concat.to(device)
            fc_concat = torch.sin(fc_concat)

            fc_concat = fc_concat.cpu().detach().squeeze().numpy()
            depth_sample = depth_sample.cpu().detach().squeeze().numpy()
            fc_concat_trans = np.transpose(fc_concat)

            new_a = np.dot(fc_concat_trans, fc_concat) + 0.001*np.identity(final_layer, dtype=np.float32)
            new_b = np.dot(fc_concat_trans, depth_sample)
            w = linalg.solve(new_a, new_b)

            # predict depth using W
            w = torch.from_numpy(w.reshape(final_layer, 1)).to(device)
            feature_map = feature_map.to(device)
            feature_map = feature_map.reshape(1, feature_map_layers, 224*224)
            index_list = np.array([[i, j] for i in range(224) for j in range(224)]).transpose(1, 0)
            index_list = torch.from_numpy(index_list).unsqueeze(0).to(device)
            
            fc_pic_in = torch.cat((index_list, feature_map), axis=1).squeeze()
            fc_pic_in = torch.transpose(fc_pic_in, 1, 0)

            fc_pic_out = base_func(fc_pic_in)
            
            predict_depth = torch.matmul(fc_pic_out, w).reshape(1, 224, 224).unsqueeze(0)

            loss = loss_function(predict_depth, disp_gt)

            writer1.add_scalar('loss', loss.item(), global_step=i+total_pics*epoch)

            disp_gt = disp_gt.squeeze().cpu().detach().numpy()
            disp_color = cv2.applyColorMap(disp_gt.astype(np.uint8), 4)
            predict_depth = predict_depth.squeeze().cpu().detach().numpy()
            predict_color = cv2.applyColorMap(predict_depth.astype(np.uint8), 4)

            if (i % 100 == 0):
                writer1.add_image('groundtruth and out', 
                                    np.concatenate((disp_color, predict_color), axis=1, out=None),
                                    global_step=i+total_pics*epoch,
                                    dataformats='HWC')

            base_func_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_func.parameters(), 10)
            base_func_optimizer.step()

            print('Epoch:'+str(epoch)+' Pic:'+str(i)+" Processing a single picture using: %.2f s"%(time.time()-time_start))
            print("---------Single points loss: %.2f" % loss.item())
            

if __name__ == '__main__':
    train_main()
