import torch
import math
from torchvision import models
from torch import nn

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

class BasicBlock(nn.Module):
    """Basic Block for Attention
    Note: This class is modified from  https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


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
        self.init_upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.init_unet4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.init_upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.init_unet5 = nn.Sequential(
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
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.finetune_output = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
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
        # self.finetune_unet1_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU()
        # )
        # self.finetune_unet2_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU()
        # )
        # self.finetune_unet3_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU()
        # )
        # self.finetune_unet4_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU()
        # )
        # self.finetune_upsample1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # self.finetune_unet5_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU()
        # )
        # self.finetune_upsample2_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # self.finetune_unet6_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU()
        # )
        # self.finetune_upsample3_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        # self.finetune_unet7_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU()
        # )

        # self.finetune_compress_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU()
        # )
        # self.finetune_output_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU()
        # )
        #################################

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

        up_1 = self.init_upsample1(out3)
        up_2 = self.init_unet4(torch.cat((up_1, out2), dim=1))
        up_3 = self.init_upsample2(up_2)
        out = self.init_unet5(torch.cat((up_3, out1), dim=1))

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

        return encoded

    def get_pose_channel(self, size_h, size_w):
        x = torch.ones((size_h, size_w)).to(device)
        x_range = torch.arange(0, size_h).unsqueeze(1).to(device, dtype=torch.float32)
        x = x * x_range / size_h
        x = x.unsqueeze(0).unsqueeze(0)

        y = torch.ones((size_h, size_w)).to(device)
        y_range = torch.arange(0, size_w).unsqueeze(0).to(device, dtype=torch.float32)
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
        out = self.finetune_output(out)

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

        return out


    def forward(self, scale8pic, scale4pic, scale2pic, depth_sample_whole):
        reso1_feature_map = self.unet_init_process(scale8pic)
        w, loss1 = self.get_w(reso1_feature_map, depth_sample_whole)
        size_h = reso1_feature_map.size()[2]
        size_w = reso1_feature_map.size()[3]
        reso1_index = self.get_pose_channel(size_h, size_w)
        
        reso1_feature_map = torch.cat((reso1_feature_map, reso1_index), dim=1)
        feature_map_layers = reso1_feature_map.size()[1]

        feature_map_reshape = reso1_feature_map.reshape(feature_map_layers, size_h*size_w).permute(1, 0)
        fc_out = self.fc_process(feature_map_reshape)
        predict_init_depth = torch.matmul(fc_out, w).reshape(1, 1, size_h, size_w)

        # finetune
        depth_upsample1 = nn.functional.interpolate(predict_init_depth, self.reso_scale4, mode='bilinear', align_corners=True)
        predict_depth_2 = self.finetune_process_2(scale4pic, depth_upsample1)

        depth_upsample2 = nn.functional.interpolate(predict_depth_2, self.reso_scale2, mode='bilinear', align_corners=True)
        predict_depth = self.finetune_process_2(scale2pic, depth_upsample2)

        return loss1, predict_depth, predict_depth_2, predict_init_depth

