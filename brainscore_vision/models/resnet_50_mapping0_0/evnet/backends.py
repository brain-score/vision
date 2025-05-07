import torch
import torchvision
from torch import nn
import torch.nn.functional as F


# Based on the implementation in
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) #
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True) # inplace=True
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, backend=False, in_channels=3, option='B'):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.option = option

        init_planes = self.inplanes = 16 if option=='A' else 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential()
        self.bn1 = nn.Sequential()
        self.relu = nn.Sequential()
        self.maxpool = nn.Sequential()
        self.layer1 = self._make_layer(block, init_planes, layers[0])
        self.layer2 = self._make_layer(block, init_planes*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, init_planes*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = nn.Sequential()

        if not backend:
            self.conv1 = nn.Conv2d(
                in_channels, init_planes, kernel_size=(3 if option=='A' else 7),
                stride=(1 if option=='A' else 2), padding=(1 if option=='A' else 3),
                bias=False
                )
            self.bn1 = norm_layer(init_planes)
            self.relu = nn.ReLU(inplace=True)
        
        last_factor = 4
        if option=='B':
            last_factor = 8
            self.layer4 = self._make_layer(
                block, init_planes*8, layers[3], stride=2,
                dilate=replace_stride_with_dilation[2]
                )
            if not backend:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear((last_factor * init_planes * block.expansion), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.option == 'A':
                downsample = lambda x: F.pad(
                    x[:, :, ::stride, ::stride],
                    (0, 0, 0, 0, 0, (planes*block.expansion)//2),
                    "constant", 0
                    )
            if self.option == 'B':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def get_resnet(
    in_channels:int,
    num_classes:int,
    layers:int=18,
    backend:bool=False,
    ):
    """Returns ResNet architecture.

    :param in_channels (int): number of input channels
    :param num_classes (int): number of classes in the classifier
    :param layers (int, optional): number of architecture layers
    :param backend (bool, optional): whether to remove the first block of the architecture
    :param tiny (bool, optional): whether to employ Tiny ImageNet adaptation (64px/2deg input)
    :return: backend model, number of backend in channels
    """
    option_A_layer_nums = [20, 32, 44, 56]
    option_B_layer_nums = [18, 34, 50, 101, 152]

    assert layers in option_A_layer_nums + option_B_layer_nums, "Invalid number of layers."
    
    bottleneck_layer_nums = [50, 101, 152]

    layer_list = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        20: [3, 3, 3],
        32: [5, 5, 5],
        44: [7, 7, 7],
        56: [9, 9, 9],
    }

    block = Bottleneck if layers in bottleneck_layer_nums else BasicBlock
    option = 'A' if layers in option_A_layer_nums else 'B'

    model = ResNet(
        block, layer_list[layers], num_classes=num_classes, backend=backend,
        in_channels=in_channels, option=option
        )
    
    if backend:
        model.in_channels = model.layer1[0].conv1.in_channels
    
    if layers == 18:
        model.conv1.stride = (1, 1)
    
    return model


def get_vgg(
    in_channels:int,
    num_classes:int,
    layers:int=16,
    backend:bool=False,
    tiny:bool=False,
    ):
    """Returns VGG16 or VGG19 backend.

    :param num_classes (int): number of classes in the classifier
    :param in_channels (float): number of input channels
    :param layers (int, optional): number of architecture layers
    :param backend (bool, optional): whether to remove the first block of the architecture
    :param tiny (bool, optional): whether to employ Tiny ImageNet adaptation (64px/2deg input)
    :return: backend model, number of backend in channels
    """
    assert layers in [16, 19]
    backend = torchvision.models.vgg16() if layers == 16 else torchvision.models.vgg19()
    backend.classifier[-1] = nn.Linear(backend.classifier[-1].in_features, num_classes)
    if tiny:
        backend.features = nn.Sequential(*list(backend.features[:-1]))
        backend.classifier[0] = nn.Linear(in_features=25088, out_features=2048, bias=True)
        backend.classifier[3] = nn.Linear(in_features=2048, out_features=2048, bias=True)
        backend.classifier[6] = nn.Linear(in_features=2048, out_features=200, bias=True)
    if backend:
        backend.features = nn.Sequential(
                *list(backend.features[2:])
                )
        backend.in_channels = backend.features[2].in_channels
    else:
        backend.features[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=backend.features[0].out_channels,
            kernel_size=backend.features[0].kernel_size,
            stride=backend.features[0].stride,
            padding=backend.features[0].padding,
            bias=True
            )
        weight = torch.zeros_like(conv1.weight.data)
        weight[:, :min(in_channels, backend_in_channels), :, :] =\
            backend.conv1.weight.data[:, :min(in_channels, backend_in_channels), :, :]
        if weight.size(1) > backend.conv1.weight.data.size(1):
            nn.init.kaiming_normal_(weight[:, in_channels:, :, :], mode="fan_out", nonlinearity="relu")
        conv1.bias.data = backend.features[0].bias.data
        conv1.weight.data = weight
        backend.features[0] = conv1
    
    return backend
