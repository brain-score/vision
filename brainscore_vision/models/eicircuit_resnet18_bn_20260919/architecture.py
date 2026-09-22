from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, features: nn.Module, readout: nn.Module):
        super().__init__()
        self.features = features
        self.readout = readout

    def forward(self, input: Tensor) -> Tensor:
        return self.readout(self.features(input))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        output = self.relu(self.bn1(self.conv1(input)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(input)
        return self.relu(output + identity)


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


class ANNVanillaResNetFeature(nn.Module):
    def __init__(self, in_channels: int = 3, feature_config: list[int] | tuple[int, ...] = (2, 2, 2, 2),
                 bottleneck: bool = False, stem: str = "imagenet"):
        super().__init__()
        if bottleneck or stem != "imagenet" or tuple(feature_config) != (2, 2, 2, 2):
            raise ValueError("This export contains ImageNet ResNet-18 only")
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(_conv1x1(self.inplanes, channels, stride),
                                       nn.BatchNorm2d(channels))
        layers = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        output = self.maxpool(self.relu(self.bn1(self.conv1(input))))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.flatten(self.avgpool(output))


class _AbsConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups,
                                                kernel_size, kernel_size))

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight.abs(), None, self.stride, self.padding,
                        1, self.groups)


class FeedforwardInhibitionBlockNoDivGainBalance(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inh_channels: int,
                 activation: nn.Module, kernel_size: int, stride: int = 1,
                 padding: int = 0, inh2exc_coordinate_exponent: float = 0.0):
        super().__init__()
        self.exc2exc = _AbsConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.exc2inh = _AbsConv2d(in_channels, inh_channels, kernel_size, stride, padding)
        self.inh2exc_sub = _AbsConv2d(inh_channels, out_channels, 1,
                                      groups=inh_channels)
        self.exc_neuron = activation
        self.inh2exc_coordinate_exponent = inh2exc_coordinate_exponent
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self.register_buffer("exc2exc_scale", torch.ones(()))
        self.register_buffer("inh2exc_scale", torch.ones(()))
        self.gain_raw = nn.Parameter(torch.zeros(()))
        self.balance_raw = nn.Parameter(torch.zeros(()))
        self.register_buffer("initial_gain", torch.ones(()))
        self.register_buffer("initial_balance_logit", torch.zeros(()))
        self.register_buffer("balance_coordinate_scale", torch.ones(()))

    @property
    def gain(self) -> Tensor:
        return self.initial_gain * F.softplus(self.gain_raw) / self.gain_raw.new_tensor(2.0).log()

    @property
    def balance(self) -> Tensor:
        return torch.tanh(self.initial_balance_logit +
                          self.balance_coordinate_scale * self.balance_raw)

    @property
    def effective_exc_scale(self) -> Tensor:
        return self.gain * (1 + self.balance)

    @property
    def effective_inh_scale(self) -> Tensor:
        return self.gain * (1 - self.balance)

    def forward(self, input: Tensor) -> Tensor:
        exc_current = self.effective_exc_scale * self.exc2exc(input)
        inh_current = self.effective_inh_scale * self.inh2exc_sub(self.exc2inh(input))
        return self.exc_neuron(exc_current - inh_current + self.bias)


def _ei_conv(in_channels: int, out_channels: int, inh_ratio: float,
             activation: nn.Module, kernel_size: int, stride: int = 1,
             padding: int = 0, exponent: float = 0.0) -> FeedforwardInhibitionBlockNoDivGainBalance:
    return FeedforwardInhibitionBlockNoDivGainBalance(
        in_channels, out_channels, int(out_channels * inh_ratio), activation,
        kernel_size, stride, padding, exponent)


class EIBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, inh_ratio: float,
                 stride: int = 1, downsample: nn.Module | None = None,
                 residual_scale: float = 1.0):
        super().__init__()
        self.conv1 = _ei_conv(in_channels, channels, inh_ratio, nn.ReLU(), 3, stride, 1)
        self.conv2 = _ei_conv(channels, channels, inh_ratio, nn.Identity(), 3, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.residual_scale = residual_scale

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        output = self.conv2(self.conv1(input))
        if self.downsample is not None:
            identity = self.downsample(input)
        return self.relu(self.residual_scale * output + identity)


class ANNEIResNetFeatureNoDivGainBalance(nn.Module):
    def __init__(self, in_channels: int = 3, inh_neuron_ratio: float = 0.25,
                 feature_config: list[int] | tuple[int, ...] = (2, 2, 2, 2),
                 bottleneck: bool = False, grouped_inh2exc: bool = True,
                 inh2exc_coordinate_exponent: float = 0.0,
                 stem: str = "imagenet", residual_scale: float = 1.0):
        super().__init__()
        if bottleneck or stem != "imagenet" or tuple(feature_config) != (2, 2, 2, 2):
            raise ValueError("This export contains ImageNet EI ResNet-18 only")
        if not grouped_inh2exc or inh2exc_coordinate_exponent != 0.0:
            raise ValueError("This export contains grouped EI with exponent zero only")
        self.inplanes = 64
        self.inh_neuron_ratio = inh_neuron_ratio
        self.grouped_inh2exc = grouped_inh2exc
        self.inh2exc_coordinate_exponent = inh2exc_coordinate_exponent
        self.residual_scale = residual_scale
        self.conv1 = _ei_conv(in_channels, 64, inh_neuron_ratio, nn.ReLU(), 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = _ei_conv(self.inplanes, channels, self.inh_neuron_ratio,
                                  nn.Identity(), 1, stride)
        layers = [EIBasicBlock(self.inplanes, channels, self.inh_neuron_ratio,
                               stride, downsample, self.residual_scale)]
        self.inplanes = channels
        layers.extend(EIBasicBlock(channels, channels, self.inh_neuron_ratio,
                                   residual_scale=self.residual_scale)
                      for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        output = self.maxpool(self.conv1(input))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.flatten(self.avgpool(output))


def build_bn_resnet18(num_classes: int = 1000) -> CNN:
    return CNN(ANNVanillaResNetFeature(), nn.Linear(512, num_classes))


def build_ei_resnet18(num_classes: int = 1000) -> CNN:
    return CNN(ANNEIResNetFeatureNoDivGainBalance(residual_scale=0.3535533905932738),
               nn.Linear(512, num_classes))


def build_model(variant: str, num_classes: int = 1000) -> CNN:
    if variant == "bn":
        return build_bn_resnet18(num_classes)
    if variant == "ei":
        return build_ei_resnet18(num_classes)
    raise ValueError(f"Unknown variant: {variant}")
