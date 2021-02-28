import torch
from torch import nn
from torchvision import models

class Resnet(torch.nn.Module):
    def __init__(self, resolution):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.upsample = torch.nn.Upsample(size=(resolution,resolution), mode="bilinear", align_corners=True)

    def forward(self, input):
        output = self.resnet.conv1(input)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        
        output = self.resnet.layer1(output)
        temp1 = output
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)
        output = self.resnet.avgpool(output)

        temp1_upsample = self.upsample(temp1)

        # feature_map = torch.cat((temp1_upsample, input), dim=1)

        return temp1_upsample


class FC_basefunction(nn.Module):
    def __init__(self, final_layer, fc_in_layer):
        super(FC_basefunction, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=fc_in_layer),
            nn.Tanh()
        )
        ############################################################
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=fc_in_layer, out_features=512),
            nn.Tanh()
        )
        #############################################################
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.Tanh()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.Tanh()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.Tanh()
        )
        #############################################################
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=512, out_features=final_layer),
        )
        #############################################################


    def forward(self, input):
        hidden_1_out = self.layer1(input)
        hidden_2_out = self.layer2(hidden_1_out)
        hidden_3_out = self.layer3(hidden_2_out)

        hidden_3_out = hidden_3_out + input
        hidden_4_out = self.layer4(hidden_3_out)

        hidden_5_out = self.layer5(hidden_4_out)
        hidden_6_out = self.layer6(hidden_5_out)
        hidden_7_out = self.layer6(hidden_6_out)

        hidden_7_out = hidden_7_out + hidden_4_out
        out = self.layer7(hidden_7_out)

        return out
