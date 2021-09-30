import numpy as np 
import torch 
import torch.nn as nn 


class UpsamplingNet(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(UpsamplingNet, self).__init__()

        self.block1 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 3,
                          stride=1, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, 1,
                          stride=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(size=(365, 590), mode='bilinear',
                              align_corners=True))

        self.block2 = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, 3,
                          stride=1, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, 1,
                          stride=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(size=(730, 1180), mode='bilinear',
                              align_corners=True))

        self.block3 = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, 3,
                          stride=1, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, 1,
                          stride=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(size=(1460, 2360), mode='bilinear',
                              align_corners=True))                              

        self.out_block = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, 3,
                          stride=1, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_ch, out_ch, 1,
                          stride=1))   

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out_block(x)
        return x


if __name__ == "__main__":
    upsampling = UpsamplingNet(1, 32, 1)

    x = torch.zeros(1, 1, 181, 294)

    x_up = upsampling(x)

    print(x_up.shape)

