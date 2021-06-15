import torch
import math
import time
import numpy as np
from torchvision import models
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRPNet(nn.Module):
    def __init__(self):
        super(DRPNet, self).__init__()
        self.unet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.unet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.unet3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.unet4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # up sample
        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.unet_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.unet_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.unet_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        self.add_channel = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )

    def forward(self, input):
        output1 = self.unet1(input) # n*32*480*640
        output2 = self.unet2(self.maxpool(output1)) # n*64*240*320
        output3 = self.unet3(self.maxpool(output2)) # n*128*120*160
        output4 = self.unet4(self.maxpool(output3)) # n*256*60*80

        up_1_in = self.upsample1(output4) # n*128*120*160
        up_1_in = torch.cat((up_1_in, output3), dim=1) # n*256*120*160
        up_1_out = self.unet_up_1(up_1_in) # n*128*120*160

        up_2_in = self.upsample2(up_1_out) # n*64*240*320
        up_2_in = torch.cat((up_2_in, output2), dim=1) # n*128*240*320
        up_2_out = self.unet_up_2(up_2_in) # n*64*240*320

        up_3_in = self.upsample3(up_2_out) # n*32*480*640
        up_3_in = torch.cat((up_3_in, output1), dim=1) # n*64*480*640
        up_3_out = self.unet_up_3(up_3_in) # N,32,480,640

        return self.add_channel(up_3_out) # N,256,H,W


class FC_basefunction(nn.Module):
    def __init__(self, final_layer, fc_in_layer):
        super(FC_basefunction, self).__init__()
        ############################################################
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.BatchNorm1d(fc_in_layer),
            nn.Tanh(),

            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.BatchNorm1d(fc_in_layer),
            nn.Tanh(),

            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.BatchNorm1d(fc_in_layer),
            nn.Tanh()
        )
        ############################################################
        self.layer1_last = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        ############################################################
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        #############################################################
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=final_layer),
            nn.BatchNorm1d(final_layer)
        )

    def forward(self, input):
        hidden_1_out = self.layer1(input)
        hidden_1_out = hidden_1_out + input
        hidden_1_out = self.layer1_last(hidden_1_out)

        hidden_2_out = self.layer2(hidden_1_out)
        hidden_2_out = hidden_2_out + hidden_1_out

        out = self.layer3(hidden_2_out)
        out = torch.sin(out)

        return out

# class BasicBlock(nn.Module):
#     """Basic Block for Attention
#     Note: This class is modified from  https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
#     """
#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class PyramidDepthEstimation(nn.Module):
    def __init__(self, reso_scale2, reso_scale4, reso_scale8, resolution_raw, sample_num):
        super(PyramidDepthEstimation, self).__init__()
        self.reso_scale2 = reso_scale2
        self.reso_scale4 = reso_scale4
        self.reso_scale8 = reso_scale8
        self.resolution_raw = resolution_raw
        self.sample_num = sample_num

        self.init_unet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.init_unet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.init_unet3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.init_unet4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        self.init_upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.init_unet5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.init_upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.init_unet6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.init_upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.init_unet7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        self.add_channel = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #################################
        self.finetune_unet1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.finetune_unet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.finetune_unet3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.finetune_unet4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.finetune_upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.finetune_unet5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.finetune_upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.finetune_unet6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.finetune_upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.finetune_unet7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.finetune_compress = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        # self.finetune_output = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU()
        # )
        #################################

        #################################
        self.finetune_res1 = BasicBlock(4, 12)
        self.finetune_res2 = BasicBlock(12, 24)
        self.finetune_res3 = BasicBlock(24, 48)
        self.finetune_res4 = BasicBlock(48, 96)
        self.finetune_res5 = BasicBlock(96, 48)
        self.finetune_res6 = BasicBlock(48, 24)
        self.finetune_res7 = BasicBlock(24, 12)
        self.finetune_res8 = BasicBlock(12, 1)

        # self.resnet_output = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(3),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(3),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU()
        # )

        self.fc_net1 = nn.Sequential(
            nn.Linear(in_features=276, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=276),
            nn.BatchNorm1d(276),
            nn.Tanh()
        )
        self.fc_net1_last = nn.Sequential(
            nn.Linear(in_features=276, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        self.fc_net2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),

            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        self.fc_net2_last = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256)
        )
        #################################


    def unet_init_process(self, scale8pic):
        out1 = self.init_unet1(scale8pic)

        out2 = self.maxpool(out1)
        out2 = self.init_unet2(out2)

        out3 = self.maxpool(out2)
        out3 = self.init_unet3(out3)

        out4 = self.maxpool(out3)
        out4 = self.init_unet4(out4)

        up_1 = self.init_upsample1(out4)
        temp_1 = self.init_unet5(torch.cat((up_1, out3), dim=1))
        up_2 = self.init_upsample2(temp_1)
        temp_2 = self.init_unet6(torch.cat((up_2, out2), dim=1))
        up_3 = self.init_upsample3(temp_2)
        out = self.init_unet7(torch.cat((up_3, out1), dim=1))

        out = self.add_channel(out)

        return out  # n x 256 x 60 x 80


    def fc_process(self, input):
        temp1 = self.fc_net1(input)
        temp1 = temp1 + input
        temp1 = self.fc_net1_last(temp1)

        temp2 = self.fc_net2(temp1)
        temp2 = temp2 + temp1
        temp2 = self.fc_net2_last(temp2)

        return torch.sin(temp2)

    def position_encoding(self, x, y):
        encoded = torch.cat((x, x, y, y), dim=1)
        for i in range(4):
            encoded = torch.cat((encoded, 
                                torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
                                torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

            # encoded = torch.cat((encoded, x, x, y, y), dim=1)

        return encoded

    def get_pose_channel(self, size_h, size_w, add_h=None, add_w=None):
        x = torch.ones((size_h, size_w)).to(device)
        x_range = torch.arange(0, size_h).unsqueeze(1).to(device, dtype=torch.float32)
        if (add_h):
            x_range = torch.arange(1, size_h+1).unsqueeze(1).to(device, dtype=torch.float32)
        x = x * x_range / size_h
        x = x.unsqueeze(0).unsqueeze(0)

        y = torch.ones((size_h, size_w)).to(device)
        y_range = torch.arange(0, size_w).unsqueeze(0).to(device, dtype=torch.float32)
        if (add_w):
            y_range = torch.arange(1, size_w+1).unsqueeze(0).to(device, dtype=torch.float32)
        y = y * y_range / size_w
        y = y.unsqueeze(0).unsqueeze(0)

        pose_channel = self.position_encoding(x, y)  # n * 20 * 60 * 80

        return pose_channel


    def get_w(self, feature_map, depth_sample_whole):
        size_h = feature_map.size()[2]
        size_w = feature_map.size()[3]

        sample_mask = depth_sample_whole >= 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask)
        depth_sample = depth_sample.reshape(self.sample_num, 1)

        pose_channel = self.get_pose_channel(size_h, size_w)

        feature_map_with_pose = torch.cat((feature_map, pose_channel), dim=1)
        feature_map_layers = feature_map_with_pose.size()[1]
        
        feature_map_sample = torch.masked_select(feature_map_with_pose, sample_mask)
        feature_map_sample = feature_map_sample.reshape(feature_map_layers, self.sample_num)
        feature_map_sample = feature_map_sample.permute(1, 0)

        fc_out = self.fc_process(feature_map_sample)
        ###############
        fc_concat_trans = fc_out.permute(1, 0)
        new_a = torch.matmul(fc_concat_trans, fc_out)
        new_b = torch.matmul(fc_concat_trans, depth_sample)
        w = torch.matmul(torch.inverse(new_a + 0.01*torch.eye(new_a.size()[0]).to(device)), new_b) # final_layers * 1
        ###############
        predect_sample_depth = torch.matmul(fc_out, w)
        loss_1 = torch.nn.functional.l1_loss(predect_sample_depth, depth_sample)

        return w, loss_1


    def finetune_process(self, pic, depth):
        input = torch.cat((pic, depth), dim=1)
        out1 = self.finetune_unet1(input)
        out2 = self.finetune_unet2(self.maxpool(out1))
        out3 = self.finetune_unet3(self.maxpool(out2))
        deepest = self.finetune_unet4(self.maxpool(out3))

        up_1 = self.finetune_upsample1(deepest)
        up_1 = self.finetune_unet5(torch.cat((up_1, out3), dim=1))
        up_2 = self.finetune_upsample2(up_1)
        up_2 = self.finetune_unet6(torch.cat((up_2, out2), dim=1))
        up_3 = self.finetune_upsample3(up_2)
        up_3 = self.finetune_unet7(torch.cat((up_3, out1), dim=1))

        out = self.finetune_compress(up_3)
        out = out + depth
        # out = self.finetune_output(out)

        return out

    def finetune_process_2(self, pic, depth):
        input = torch.cat((pic, depth), dim=1)
        out = self.finetune_res1(input)
        out = self.finetune_res2(out)
        out = self.finetune_res3(out)
        out = self.finetune_res4(out)
        out = self.finetune_res5(out)
        out = self.finetune_res6(out)
        out = self.finetune_res7(out)
        out = self.finetune_res8(out)

        out = out + depth
        # out = self.resnet_output(out)

        return out

    def gradient(self, x):
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

        return dx, dy # dw, dh


    def forward(self, scale4pic, scale2pic, depth_sample_whole, use_dxdy=False):
        reso1_feature_map = self.unet_init_process(scale4pic)
        w, loss1 = self.get_w(reso1_feature_map, depth_sample_whole)
        
        size_h = reso1_feature_map.size()[2]
        size_w = reso1_feature_map.size()[3]
        reso1_index = self.get_pose_channel(size_h, size_w)
        # reso1_index.requires_grad = True

        reso1_feature_map_pose = torch.cat((reso1_feature_map, reso1_index), dim=1)

        feature_map_layers = reso1_feature_map_pose.size()[1]
        feature_map_reshape = reso1_feature_map_pose.reshape(feature_map_layers, size_h*size_w).permute(1, 0)
        fc_out = self.fc_process(feature_map_reshape)
        predict_init_depth = torch.matmul(fc_out, w).reshape(1, 1, size_h, size_w)

        #########################################
        if (use_dxdy):
            dw, dh = self.gradient(predict_init_depth)
            reso1_index_add_h = self.get_pose_channel(size_h, size_w, add_h=True)
            reso1_feature_map_add_h = torch.cat((reso1_feature_map, reso1_index_add_h), dim=1)
            feature_map_reshape_add_h = reso1_feature_map_add_h.reshape(feature_map_layers, size_h*size_w).permute(1, 0)
            fc_out_add_h = self.fc_process(feature_map_reshape_add_h)

            reso1_index_add_w = self.get_pose_channel(size_h, size_w, add_w=True)
            reso1_feature_map_add_w = torch.cat((reso1_feature_map, reso1_index_add_w), dim=1)
            feature_map_reshape_add_w = reso1_feature_map_add_w.reshape(feature_map_layers, size_h*size_w).permute(1, 0)
            fc_out_add_w = self.fc_process(feature_map_reshape_add_w)

            delta_h = fc_out_add_h - fc_out
            delta_w = fc_out_add_w - fc_out
            depth_delta_h = torch.matmul(delta_h, w).reshape(1, 1, size_h, size_w)
            depth_delta_w = torch.matmul(delta_w, w).reshape(1, 1, size_h, size_w)

            delta_h_loss = torch.nn.functional.l1_loss(depth_delta_h, dh)
            delta_w_loss = torch.nn.functional.l1_loss(depth_delta_w, dw)
        else:
            delta_h_loss = None
            delta_w_loss = None
        #########################################

        # finetune
        depth_finetune_1 = self.finetune_process(scale4pic, predict_init_depth)
        depth_upsample1 = nn.functional.interpolate(depth_finetune_1, self.reso_scale2, mode='bilinear', align_corners=True)
        depth_finetune_2 = self.finetune_process(scale2pic, depth_upsample1)


        return loss1, delta_h_loss, delta_w_loss, depth_finetune_1, depth_finetune_2, predict_init_depth


def doubleconv2d(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.Tanh()
            nn.LeakyReLU()
        )

def doublefc1d(in_channels, out_channels):
    return nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels)
        )

def triblefc1d(in_channels):
    return nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Tanh(),
            nn.Linear(in_features=in_channels, out_features=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Tanh(),
            nn.Linear(in_features=in_channels, out_features=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Tanh()
        )

class Siren3dEstimation(nn.Module):
    def __init__(self, sample_num):
        super(Siren3dEstimation, self).__init__()
        self.resolution = (228, 304)
        self.sample_num = sample_num

        #####################################################################
        self.first_unet_down1 = doubleconv2d(3, 32)
        self.first_unet_down2 = doubleconv2d(32, 64)
        self.first_unet_down3 = doubleconv2d(64, 128)
        self.first_unet_down4 = doubleconv2d(128, 256)
        self.first_unet_down5 = doubleconv2d(256, 512)

        self.first_unet_up1 = doubleconv2d(512+256, 256)
        self.first_unet_up2 = doubleconv2d(256+128, 128)
        self.first_unet_up3 = doubleconv2d(128+64, 64)
        self.first_unet_up4 = doubleconv2d(64+32, 32)

        self.add_channel1 = doubleconv2d(32, 64)
        self.add_channel2 = doubleconv2d(64, 128)
        self.add_channel3 = doubleconv2d(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #####################################################################
        # self.fc_net1 = triblefc1d(296)
        # self.fc_net2 = triblefc1d(512)
        # self.fc_net1_change = nn.Sequential(
        #     nn.Linear(in_features=296, out_features=512),
        #     nn.BatchNorm1d(512),
        #     nn.Tanh()
        # )
        # self.fc_net2_change = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.BatchNorm1d(256)
        # )
        #####################################################################
        self.fc_net1 = nn.Sequential(
            nn.Linear(in_features=296, out_features=296),
            nn.BatchNorm1d(296),
            nn.Tanh(),
            nn.Linear(in_features=296, out_features=296),
            nn.BatchNorm1d(296),
            nn.Tanh(),
            nn.Linear(in_features=296, out_features=296)
        )
        self.fc_net2 = nn.Sequential(
            nn.Linear(in_features=296, out_features=256),
            nn.BatchNorm1d(256)
        )
        self.fc_bn = nn.BatchNorm1d(296)
        self.fc_act = nn.Tanh()
        #####################################################################

    def unet_init_process(self, rgb):
        unet1 = self.first_unet_down1(rgb)
        x = self.maxpool(unet1)

        unet2 = self.first_unet_down2(x)
        x = self.maxpool(unet2)

        unet3 = self.first_unet_down3(x)
        x = self.maxpool(unet3)

        unet4 = self.first_unet_down4(x)
        x = self.maxpool(unet4)

        unet5 = self.first_unet_down5(x)
        # x = self.maxpool(unet5)

        x = nn.functional.interpolate(unet5, (unet4.size()[2], unet4.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet4), dim=1)
        x = self.first_unet_up1(x)

        x = nn.functional.interpolate(x, (unet3.size()[2], unet3.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet3), dim=1)
        x =self.first_unet_up2(x)

        x = nn.functional.interpolate(x, (unet2.size()[2], unet2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet2), dim=1)
        x = self.first_unet_up3(x)

        x = nn.functional.interpolate(x, (unet1.size()[2], unet1.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet1), dim=1)
        x = self.first_unet_up4(x)

        x = self.add_channel1(x)
        x = self.add_channel2(x)
        x = self.add_channel3(x)

        return x


    def fc_process(self, input):
        # res1 = self.fc_net1(input)
        # res1 = res1 + input
        # res1 = self.fc_net1_change(res1)

        # res2 = self.fc_net2(res1)
        # res2 = res2 + res1
        # res2 = self.fc_net2_change(res2)
        # res2 = torch.sin(res2)
        #####################################################################
        res1 = self.fc_net1(input)
        res1 = res1 + input
        res1 = self.fc_bn(res1)
        res1 = self.fc_act(res1)

        res2 = self.fc_net2(res1)
        res2 = torch.sin(res2)
        # res2 = self.fc_act(res2)

        return res2

    def position_encoding(self, index_list):
        x, y = torch.split(index_list, 1, dim=1)
        encoded = torch.cat((x, x, y, y), dim=1)
        # encoded = torch.cat((encoded, encoded, encoded, encoded, encoded), dim=1)
        for i in range(9):
            encoded = torch.cat((encoded, 
                                torch.sin((2**(i))*math.pi*x), torch.cos((2**(i))*math.pi*x),
                                torch.sin((2**(i))*math.pi*y), torch.cos((2**(i))*math.pi*y)), dim=1)

            # encoded = torch.cat((encoded, 
            #                     torch.sin((0.5+i*0.1875)*math.pi*x), torch.cos((0.5+i*0.1875)*math.pi*x),
            #                     torch.sin((0.5+i*0.1875)*math.pi*y), torch.cos((0.5+i*0.1875)*math.pi*y)), dim=1)

        return encoded # n * 20 * h * w

    def get_w(self, feature_map, depth_sample_whole, index_list):
        sample_mask = depth_sample_whole >= 0
        depth_sample = torch.masked_select(depth_sample_whole, sample_mask)
        depth_sample = depth_sample.reshape(self.sample_num, 1)

        pose_channel = self.position_encoding(index_list)

        feature_map_with_pose = torch.cat((feature_map, pose_channel), dim=1)
        feature_map_layers = feature_map_with_pose.size()[1]
        
        feature_map_sample = torch.masked_select(feature_map_with_pose, sample_mask)
        feature_map_sample = feature_map_sample.reshape(feature_map_layers, self.sample_num)
        feature_map_sample = feature_map_sample.permute(1, 0)

        fc_out = self.fc_process(feature_map_sample)
        ###############
        fc_concat_trans = fc_out.permute(1, 0)
        new_a = torch.matmul(fc_concat_trans, fc_out)
        new_b = torch.matmul(fc_concat_trans, depth_sample)
        w = torch.matmul(torch.inverse(new_a + 0.001*torch.eye(new_a.size()[0]).to(device)), new_b) # final_layers * 1
        ###############
        predict_sample_depth = torch.matmul(fc_out, w)
        loss_1 = torch.nn.functional.l1_loss(depth_sample, predict_sample_depth)

        return w, loss_1

    # def get_pose_channel(self, size_h, size_w):
    #     x = torch.ones((size_h, size_w)).to(device)
    #     x_range = torch.arange(0, size_h).unsqueeze(1).to(device, dtype=torch.float32)
    #     x = x * x_range / torch.tensor(size_h).to(device) * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)
    #     x = x.unsqueeze(0).unsqueeze(0)

    #     y = torch.ones((size_h, size_w)).to(device)
    #     y_range = torch.arange(0, size_w).unsqueeze(0).to(device, dtype=torch.float32)
    #     y = y * y_range / torch.tensor(size_w).to(device) * torch.tensor(2.).to(device) - torch.tensor(1.).to(device)
    #     y = y.unsqueeze(0).unsqueeze(0)

    #     return torch.cat((x, y), dim=1)

    def forward(self, rgb, depth_sample_whole, index_list, index_list_copy):
        feature_map = self.unet_init_process(rgb)
        w, loss1= self.get_w(feature_map, depth_sample_whole, index_list)

        pose_channel = self.position_encoding(index_list_copy)
        feature_map_withpose = torch.cat((feature_map, pose_channel), dim=1)

        feature_map_layers = feature_map_withpose.size()[1]
        feature_map_withpose = feature_map_withpose.reshape(feature_map_layers, self.resolution[0]*self.resolution[1]).permute(1, 0)
        fc_out = self.fc_process(feature_map_withpose)
        predict_init_depth = torch.matmul(fc_out, w).reshape(1, 1, self.resolution[0], self.resolution[1])

        return loss1, predict_init_depth


class Finetune(nn.Module):
    def __init__(self):
        super(Finetune, self).__init__()
        self.finetune_down1 = doubleconv2d(4, 32)
        self.finetune_down2 = doubleconv2d(32, 64)
        self.finetune_down3 = doubleconv2d(64, 128)
        self.finetune_down4 = doubleconv2d(128, 256)
        self.finetune_down5 = doubleconv2d(256, 512)

        self.finetune_up1 = doubleconv2d(512+256, 256)
        self.finetune_up2 = doubleconv2d(256+128, 128)
        self.finetune_up3 = doubleconv2d(128+64, 64)
        self.finetune_up4 = doubleconv2d(64+32, 32)

        self.finetune_compress = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            # nn.Tanh()
            nn.LeakyReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, init_depth, rgb):
        input = torch.cat((rgb, init_depth), dim=1)

        out1 = self.finetune_down1(input)
        x = self.maxpool(out1)

        out2 = self.finetune_down2(x)
        x = self.maxpool(out2)

        out3 = self.finetune_down3(x)
        x = self.maxpool(out3)

        out4 = self.finetune_down4(x)
        x = self.maxpool(out4)

        out5 = self.finetune_down5(x)

        x = nn.functional.interpolate(out5, (out4.size()[2], out4.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out4), dim=1)
        x = self.finetune_up1(x)

        x = nn.functional.interpolate(x, (out3.size()[2], out3.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out3), dim=1)
        x = self.finetune_up2(x)

        x = nn.functional.interpolate(x, (out2.size()[2], out2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out2), dim=1)
        x = self.finetune_up3(x)

        x = nn.functional.interpolate(x, (out1.size()[2], out1.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out1), dim=1)
        x = self.finetune_up4(x)

        x = self.finetune_compress(x)
        
        out = x + init_depth

        return out


class ConvInit(nn.Module):
    def __init__(self):
        super(ConvInit, self).__init__()
        self.layer1 = doubleconv2d(1, 3)
        self.layer2 = doubleconv2d(3, 1)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = output + input

        return output


class FeatureExtractionNet(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet, self).__init__()
        self.first_unet_down1 = doubleconv2d(3, 32)
        self.first_unet_down2 = doubleconv2d(32, 64)
        self.first_unet_down3 = doubleconv2d(64, 128)
        self.first_unet_down4 = doubleconv2d(128, 256)
        self.first_unet_down5 = doubleconv2d(256, 512)

        self.first_unet_up1 = doubleconv2d(512+256, 256)
        self.first_unet_up2 = doubleconv2d(256+128, 128)
        self.first_unet_up3 = doubleconv2d(128+64, 64)
        self.first_unet_up4 = doubleconv2d(64+32, 32)

        self.add_channel1 = doubleconv2d(32, 64)
        self.add_channel2 = doubleconv2d(64, 128)
        self.add_channel3 = doubleconv2d(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)   

    def forward(self, rgb):
        unet1 = self.first_unet_down1(rgb)
        x = self.maxpool(unet1)

        unet2 = self.first_unet_down2(x)
        x = self.maxpool(unet2)

        unet3 = self.first_unet_down3(x)
        x = self.maxpool(unet3)

        unet4 = self.first_unet_down4(x)
        x = self.maxpool(unet4)

        unet5 = self.first_unet_down5(x)

        x = nn.functional.interpolate(unet5, (unet4.size()[2], unet4.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet4), dim=1)
        x = self.first_unet_up1(x)

        x = nn.functional.interpolate(x, (unet3.size()[2], unet3.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet3), dim=1)
        x =self.first_unet_up2(x)

        x = nn.functional.interpolate(x, (unet2.size()[2], unet2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet2), dim=1)
        x = self.first_unet_up3(x)

        x = nn.functional.interpolate(x, (unet1.size()[2], unet1.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet1), dim=1)
        x = self.first_unet_up4(x)

        x = self.add_channel1(x)
        x = self.add_channel2(x)
        x = self.add_channel3(x)

        return x


class FullConnected(nn.Module):
    def __init__(self):
        super(FullConnected, self).__init__()
        fc_in_layer = 296
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.BatchNorm1d(fc_in_layer),
            nn.Tanh(),

            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.BatchNorm1d(fc_in_layer),
            nn.Tanh(),

            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer)
            # nn.BatchNorm1d(fc_in_layer),
            # nn.Tanh()
        )
        ############################################################
        self.layer1_last = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        ############################################################
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(in_features=512, out_features=512)
            # nn.BatchNorm1d(512),
            # nn.Tanh()
        )
        ############################################################
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256)
            # nn.BatchNorm1d(256)
        )
        self.fc_bn_296 = nn.BatchNorm1d(296)
        self.fc_bn_512 = nn.BatchNorm1d(512)
        self.fc_act = nn.Tanh()

    def forward(self, input):
        hidden_1_out = self.layer1(input)
        hidden_1_out = hidden_1_out + input
        hidden_1_out = self.fc_bn_296(hidden_1_out)
        hidden_1_out = self.fc_act(hidden_1_out)

        hidden_1_out = self.layer1_last(hidden_1_out)

        hidden_2_out = self.layer2(hidden_1_out)
        hidden_2_out = hidden_2_out + hidden_1_out
        hidden_2_out = self.fc_bn_512(hidden_2_out)
        hidden_2_out = self.fc_act(hidden_2_out)

        out = self.layer3(hidden_2_out)
        # out = torch.sin(out)
        out = self.fc_act(out)

        return out

class FullConnected_baseline(nn.Module):
    def __init__(self):
        super(FullConnected_baseline, self).__init__()
        self.fc_net1 = nn.Sequential(
            nn.Linear(in_features=296, out_features=296),
            nn.BatchNorm1d(296),
            nn.Tanh(),
            nn.Linear(in_features=296, out_features=296),
            nn.BatchNorm1d(296),
            nn.Tanh(),
            nn.Linear(in_features=296, out_features=296)
        )
        self.fc_net2 = nn.Sequential(
            nn.Linear(in_features=296, out_features=256),
            nn.BatchNorm1d(256)
        )
        self.fc_bn = nn.BatchNorm1d(296)
        self.fc_act = nn.Tanh()

    def forward(self, input):
        res1 = self.fc_net1(input)
        res1 = res1 + input
        res1 = self.fc_bn(res1)
        res1 = self.fc_act(res1)

        res2 = self.fc_net2(res1)
        # res2 = torch.sin(res2)
        res2 = self.fc_act(res2)

        return res2


class Conv_1d(nn.Module):
    def __init__(self):
        super(Conv_1d, self).__init__()
        self.w_layers = 128
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=296, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512)
        )
        self.layer1_res = nn.Sequential(
            nn.Conv1d(in_channels=296, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.w_layers, kernel_size=1),
            nn.BatchNorm1d(self.w_layers),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.w_layers, out_channels=self.w_layers, kernel_size=1),
            nn.BatchNorm1d(self.w_layers),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.w_layers, out_channels=self.w_layers, kernel_size=1),
            nn.BatchNorm1d(self.w_layers)
        )
        self.layer2_res = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.w_layers, kernel_size=1),
            nn.BatchNorm1d(self.w_layers)
        )
        self.batch512 = nn.BatchNorm1d(512)
        self.batch256 = nn.BatchNorm1d(256)
    
    def forward(self, input):
        res1 = self.layer1(input) + self.layer1_res(input)
        # res1 = nn.Tanh()(res1)
        res1 = nn.LeakyReLU()(res1)

        res2 = self.layer2(res1) + self.layer2_res(res1)
        # res2 = nn.Tanh()(res2)
        res2 = nn.LeakyReLU()(res2)

        return res2


class FeatureExtractionNet_Change(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet_Change, self).__init__()
        self.first_unet_down1 = doubleconv2d(3, 32)
        self.first_unet_down2 = doubleconv2d(32, 64)
        self.first_unet_down3 = doubleconv2d(64, 128)
        self.first_unet_down4 = doubleconv2d(128, 256)

        self.first_unet_up1 = doubleconv2d(256+128, 128)
        self.first_unet_up2 = doubleconv2d(128+64, 64)
        self.first_unet_up3 = doubleconv2d(64+32, 32)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.add_channel = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
           
    def forward(self, rgb):
        unet1 = self.first_unet_down1(rgb)
        x = self.maxpool(unet1)

        unet2 = self.first_unet_down2(x)
        x = self.maxpool(unet2)

        unet3 = self.first_unet_down3(x)
        x = self.maxpool(unet3)

        unet4 = self.first_unet_down4(x)

        x = nn.functional.interpolate(unet4, (unet3.size()[2], unet3.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet3), dim=1)
        x = self.first_unet_up1(x)

        x = nn.functional.interpolate(x, (unet2.size()[2], unet2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet2), dim=1)
        x =self.first_unet_up2(x)

        x = nn.functional.interpolate(x, (unet1.size()[2], unet1.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, unet1), dim=1)
        x = self.first_unet_up3(x)

        x = self.add_channel(x)

        return x


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 1,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 256, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class FinetuneDistribution(nn.Module):
    def __init__(self):
        super(FinetuneDistribution, self).__init__()
        self.Mlayers = 401
        self.start = -4
        self.stop = 4
        self.fixed = np.array([i * ((self.stop-self.start) / (self.Mlayers-1)) + self.start for i in range(self.Mlayers)]).reshape(1, self.Mlayers, 1, 1)
        # self.fixed = torch.from_numpy(self.fixed).expand(1, 69312, self.Mlayers).permute(0, 2, 1).reshape(1, self.Mlayers, 228, 304)
        self.fixed = torch.from_numpy(self.fixed).to(device, dtype=torch.float32)
        self.fixed = nn.Parameter(data=self.fixed, requires_grad=False)

        self.finetune_down1 = doubleconv2d(4, 32)
        self.finetune_down2 = doubleconv2d(32, 64)
        self.finetune_down3 = doubleconv2d(64, 128)
        self.finetune_down4 = doubleconv2d(128, 256)
        self.finetune_down5 = doubleconv2d(256, 512)

        self.finetune_up1 = doubleconv2d(512+256, 256)
        self.finetune_up2 = doubleconv2d(256+128, 128)
        self.finetune_up3 = doubleconv2d(128+64, 64)
        self.finetune_up4 = doubleconv2d(64+32, 32)

        self.finetune_addlayers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=401, kernel_size=1),
            nn.BatchNorm2d(401)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, init_depth):
        input = torch.cat((rgb, init_depth), dim=1)

        out1 = self.finetune_down1(input)
        x = self.maxpool(out1)

        out2 = self.finetune_down2(x)
        x = self.maxpool(out2)

        out3 = self.finetune_down3(x)
        x = self.maxpool(out3)

        out4 = self.finetune_down4(x)
        x = self.maxpool(out4)

        out5 = self.finetune_down5(x)

        x = nn.functional.interpolate(out5, (out4.size()[2], out4.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out4), dim=1)
        x = self.finetune_up1(x)

        x = nn.functional.interpolate(x, (out3.size()[2], out3.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out3), dim=1)
        x = self.finetune_up2(x)

        x = nn.functional.interpolate(x, (out2.size()[2], out2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out2), dim=1)
        x = self.finetune_up3(x)

        x = nn.functional.interpolate(x, (out1.size()[2], out1.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat((x, out1), dim=1)
        x = self.finetune_up4(x)

        x = self.finetune_addlayers(x)
        x = self.softmax(x) # 1x401x228x304

        output = F.conv2d(x, self.fixed)

        # x = x.reshape(self.Mlayers, 69312).permute(1, 0)
        # output = torch.matmul(x, self.fixed).reshape(1, 1, 228, 304)

        return x, (output + init_depth)


# if __name__ == "__main__":
#     f_e_net = feature_extraction()
#     img = torch.rand(1, 3, 228, 304)
#     out = f_e_net(img)
#     print(out.size())