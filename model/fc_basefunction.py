import torch
from torch import nn
from torchvision import models

class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.upsample = torch.nn.Upsample(size=(224,224), mode="bilinear", align_corners=True)

    def forward(self, input):
        output = self.resnet.conv1(input)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        temp1 = output
        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)
        output = self.resnet.avgpool(output)

        temp1_upsample = self.upsample(temp1)

        feature_map = torch.cat((temp1_upsample, input), dim=1)

        return feature_map


class FC_basefunction(nn.Module):
    def __init__(self, final_layer):
        super(FC_basefunction, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=69, out_features=256),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=final_layer)
        )

    def forward(self, input):
        hidden_1_out = self.layer1(input)
        hidden_2_out = self.layer2(hidden_1_out)
        hidden_3_out = self.layer3(hidden_2_out)
        return hidden_3_out
