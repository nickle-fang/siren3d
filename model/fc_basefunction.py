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
        temp = output
        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)
        output = self.resnet.avgpool(output)
        return self.upsample(temp)


class FC_basefunction(nn.Module):
    def __init__(self):
        super(FC_basefunction, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=66, out_features=256),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64)
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(True)
        )
        # self.layer4 = nn.Sequential(
        #     nn.Linear(in_features=16, out_features=3)
        #     # nn.BatchNorm1d(3)
        # )

    def forward(self, input):
        hidden_1_out = self.layer1(input)
        hidden_2_out = self.layer2(hidden_1_out)
        hidden_3_out = self.layer3(hidden_2_out)
        # hidden_4_out = self.layer4(hidden_3_out)
        return hidden_3_out
