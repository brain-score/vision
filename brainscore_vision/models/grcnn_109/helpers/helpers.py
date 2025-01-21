import torch
import torch.nn as nn
from torch.nn import init
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define your custom model here:
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32, groups=32):

        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=groups,
                              bias=False)
            init.kaiming_normal_(conv1.weight)
            self.conv.append(nn.Sequential(conv1,
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        conv_fc = nn.Conv2d(out_channels, d, 1, bias=False)
        init.normal_(conv_fc.weight, std=0.01)
        self.fc1 = nn.Sequential(conv_fc,
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        init.normal_(self.fc2.weight, std=0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


class GRCL(nn.Module):
    def __init__(self, inplanes, planes, downsample=True, iter=3, SKconv=True, expansion=2):
        super(GRCL, self).__init__()

        self.iter = iter
        self.expansion = expansion
        # feed-forward part
        self.add_module('bn_f', nn.BatchNorm2d(inplanes))
        self.add_module('relu_f', nn.ReLU(inplace=True))
        conv_f = nn.Conv2d(inplanes, int(planes * self.expansion), kernel_size=3, stride=1, padding=1, bias=False,
                           groups=32)
        init.kaiming_normal_(conv_f.weight)
        self.add_module('conv_f', conv_f)

        self.add_module('bn_g_f', nn.BatchNorm2d(inplanes))
        self.add_module('relu_g_f', nn.ReLU(inplace=True))
        conv_g_f = nn.Conv2d(inplanes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=True,
                             groups=32)
        init.normal_(conv_g_f.weight, std=0.01)
        self.add_module('conv_g_f', conv_g_f)
        self.conv_g_r = nn.Conv2d(int(planes * self.expansion), int(planes * self.expansion), kernel_size=1, stride=1,
                                  padding=0, bias=False, groups=32)
        self.add_module('sig', nn.Sigmoid())

        # recurrent part
        for i in range(0, self.iter):
            layers = []
            layers_g_bn = []

            layers.append(nn.BatchNorm2d(planes * self.expansion))
            layers.append(nn.ReLU(inplace=True))
            conv_1 = nn.Conv2d(int(planes * self.expansion), planes, kernel_size=1, stride=1, padding=0, bias=False)
            init.kaiming_normal_(conv_1.weight)
            layers.append(conv_1)

            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))

            if SKconv:
                layers.append(SKConv(planes, planes))
            else:
                layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))

            conv_2 = nn.Conv2d(planes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
            init.kaiming_normal_(conv_2.weight)
            layers.append(conv_2)
            layers_g_bn.append(nn.BatchNorm2d(int(planes * self.expansion)))

            layers_g_bn.append(nn.ReLU(inplace=True))

            self.add_module('iter_' + str(i + 1), nn.Sequential(*layers))
            self.add_module('iter_g_' + str(i + 1), nn.Sequential(*layers_g_bn))

        self.downsample = downsample
        if self.downsample:
            self.add_module('d_bn', nn.BatchNorm2d(planes * self.expansion))
            self.add_module('d_relu', nn.ReLU(inplace=True))
            d_conv = nn.Conv2d(int(planes * self.expansion), int(planes * self.expansion), kernel_size=1, stride=1,
                               padding=0, bias=False)
            init.kaiming_normal_(d_conv.weight)
            self.add_module('d_conv', d_conv)
            self.add_module('d_ave', nn.AvgPool2d((2, 2), stride=2))

            self.add_module('d_bn_1', nn.BatchNorm2d(planes * self.expansion))
            self.add_module('d_relu_1', nn.ReLU(inplace=True))
            d_conv_1 = nn.Conv2d(int(planes * self.expansion), planes, kernel_size=1, stride=1, padding=0,
                                 bias=False)
            init.kaiming_normal_(d_conv_1.weight)
            self.add_module('d_conv_1', d_conv_1)

            self.add_module('d_bn_3', nn.BatchNorm2d(planes))
            self.add_module('d_relu_3', nn.ReLU(inplace=True))

            if SKconv:
                d_conv_3 = SKConv(planes, planes, stride=2)
                self.add_module('d_conv_3', d_conv_3)
            else:
                d_conv_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
                init.kaiming_normal_(d_conv_3.weight)
                self.add_module('d_conv_3', d_conv_3)

            d_conv_1e = nn.Conv2d(planes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
            init.kaiming_normal_(d_conv_1e.weight)
            self.add_module('d_conv_1e', d_conv_1e)

    def forward(self, x):
        # feed-forward
        x_bn = self.bn_f(x)
        x_act = self.relu_f(x_bn)
        x_s = self.conv_f(x_act)

        x_g_bn = self.bn_g_f(x)
        x_g_act = self.relu_g_f(x_g_bn)
        x_g_s = self.conv_g_f(x_g_act)

        # recurrent
        for i in range(0, self.iter):
            x_g_r = self.conv_g_r(self.__dict__['_modules']["iter_g_%s" % str(i + 1)](x_s))
            x_s = self.__dict__['_modules']["iter_%s" % str(i + 1)](x_s) * torch.sigmoid(x_g_r + x_g_s) + x_s

        if self.downsample:
            x_s_1 = self.d_conv(self.d_ave(self.d_relu(self.d_bn(x_s))))
            x_s_2 = self.d_conv_1e(
                self.d_conv_3(self.d_relu_3(self.d_bn_3(self.d_conv_1(self.d_relu_1(self.d_bn_1(x_s)))))))
            x_s = x_s_1 + x_s_2

        return x_s


class GRCNN(nn.Module):

    def __init__(self, iters, maps, SKconv, expansion, num_classes):
        """ Args:
          iters:iterations.
          num_classes: number of classes
        """
        super(GRCNN, self).__init__()
        self.iters = iters
        self.maps = maps
        self.num_classes = num_classes
        self.expansion = expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        init.kaiming_normal_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

        init.kaiming_normal_(self.conv2.weight)

        self.layer1 = GRCL(64, self.maps[0], True, self.iters[0], SKconv, self.expansion)
        self.layer2 = GRCL(self.maps[0] * self.expansion, self.maps[1], True, self.iters[1], SKconv, self.expansion)
        self.layer3 = GRCL(self.maps[1] * self.expansion, self.maps[2], True, self.iters[2], SKconv, self.expansion)
        self.layer4 = GRCL(self.maps[2] * self.expansion, self.maps[3], False, self.iters[3], SKconv, self.expansion)

        self.lastact = nn.Sequential(nn.BatchNorm2d(self.maps[3] * self.expansion), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(self.maps[3] * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def grcnn109(num_classes=1000):
    """
    Args:
      num_classes (uint): number of classes
    """
    model = GRCNN([3, 3, 22, 3], [64, 128, 256, 512], SKconv=False, expansion=4, num_classes=num_classes)
    return model


