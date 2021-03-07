from numpy.lib.arraypad import pad
from numpy.lib.index_tricks import AxisConcatenator
import torch
from torch import nn


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
