import torch
import torch.nn as nn
import torch.nn.functional as F
from vonenet import VOneNet

# Classicl residual Block as in ResNet


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, i):
        x = self.act(self.bn1(self.conv1(i)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.act(x+i)

# g_t = g_{t-1} + B(x_{t-1})
# x_t = x_{t-1} + sigma(g_t).*A(x_{t-1})


class myGRCL(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        self.A = ResBlock(in_channels, out_channels)
        self.B = ResBlock(in_channels, out_channels)
        self.T = T

    def forward(self, x, g):
        for _ in range(self.T):
            g = g + self.B(x)
            x = x + torch.sigmoid(g)*self.A(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.VOneBlock = VOneNet(model_arch=None)
        self.myGRCL1 = myGRCL(in_channels=512, out_channels=512, T=4)
        self.myGRCL2 = myGRCL(in_channels=512, out_channels=512, T=4)
        self.myGRCL3 = myGRCL(in_channels=512, out_channels=512, T=4)
        self.myGRCL4 = myGRCL(in_channels=512, out_channels=512, T=4)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.VOneBlock(x)
        x = self.myGRCL1(x, x)
        x = self.pool(x)
        x = self.myGRCL2(x, x)
        x = self.pool(x)
        x = self.myGRCL3(x, x)
        x = self.pool(x)
        x = self.myGRCL4(x, x)
        x = torch.max(x, dim=-1)[0]
        x = torch.max(x, dim=-1)[0]
        x = self.fc(x)
        return x
