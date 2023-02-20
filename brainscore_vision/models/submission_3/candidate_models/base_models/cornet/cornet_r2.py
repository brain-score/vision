from collections import OrderedDict

import math
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class CORBlock_Rec2(nn.Module):
    scale = 6

    def __init__(self, in_channels, out_channels, ntimes=1, stride=1, h=None, name=None):
        super(CORBlock_Rec2, self).__init__()

        self.name = name
        self.ntimes = ntimes
        self.stride = stride

        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.output = Identity()

        for n in range(ntimes):
            setattr(self, f'bn1_{n}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'bn2_{n}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'bn3_{n}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        inp = self.conv_first(inp)

        for n in range(self.ntimes):
            if n == 0:
                x = inp
                residual = self.shortcut_bn(self.shortcut(inp))
                inp = residual
            else:
                residual = inp + state
                x = residual

            if n == 0 and self.stride == 2:
                self.conv2.stride = (2, 2)
            else:
                self.conv2.stride = (1, 1)
            x = self.conv2(x)
            x = getattr(self, f'bn2_{n}')(x)
            x = self.relu2(x)

            x = self.conv3(x)
            x = getattr(self, f'bn3_{n}')(x)

            x += residual
            state = self.relu3(x)
            state = self.output(state)

        return x


class CORNetR2(nn.Module):
    def __init__(self, ntimes=(5, 5, 5), num_classes=1000):
        super(CORNetR2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V1 = nn.Sequential(OrderedDict([('output', Identity())]))

        self.V2 = CORBlock_Rec2(64, 128, ntimes=ntimes[0], stride=2, h=28, name='b0')
        self.V4 = CORBlock_Rec2(128, 256, ntimes=ntimes[1], stride=2, h=14, name='b1')
        self.IT = CORBlock_Rec2(256, 512, ntimes=ntimes[2], stride=2, h=7, name='b2')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def fix_state_dict_naming(state_dict):
    block_region_mapping = {'block2': 'V2', 'block3': 'V4', 'block4': 'IT'}  # no V1 because introduced identity output
    block_region_mapping = {f"module.{block}": f"module.{region}" for block, region in block_region_mapping.items()}

    def rename_module(module):
        for block, region in block_region_mapping.items():
            if module.startswith(block):
                module = region + module[len(block):]
                break
        return module

    expected_num_keys = len(state_dict)
    metadata = OrderedDict([(rename_module(module), meta) for module, meta in state_dict._metadata.items()])
    state_dict = OrderedDict([(rename_module(module), weights) for module, weights in state_dict.items()])
    state_dict._metadata = metadata
    assert len(state_dict) == expected_num_keys
    return state_dict
