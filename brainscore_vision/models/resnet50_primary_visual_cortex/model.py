import functools
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision import transforms

from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


class OpponentChannelInhibition(nn.Module):
    def __init__(self, n_channels):
        super(OpponentChannelInhibition, self).__init__()
        self.n_channels = n_channels
        channel_inds = torch.arange(n_channels, dtype=torch.float32)+1.
        channel_inds = channel_inds-channel_inds[n_channels//2]
        self.channel_inds = torch.abs(channel_inds)
        channel_distances = []
        for i in range(n_channels):
            channel_distances.append(torch.roll(self.channel_inds, i))
        self.channel_distances = nn.Parameter(torch.stack(channel_distances), requires_grad=False)
        self.sigmas = nn.Parameter(torch.rand(n_channels)+(n_channels/8), requires_grad=True)

    def forward(self, x):
        sigmas = torch.clamp(self.sigmas, min=0.5)
        gaussians = (1/(2.5066*sigmas))*torch.exp(-1*(self.channel_distances**2)/(2*(sigmas**2))) # sqrt(2*pi) ~= 2.5066
        gaussians = gaussians/torch.sum(gaussians, dim=0)
        gaussians = gaussians.view(self.n_channels,self.n_channels,1,1)
        weighted_channel_inhibition = nn.functional.conv2d(x, weight=gaussians, stride=1, padding=0)
        return x/(weighted_channel_inhibition+1)
    

class DoGConv2D_v2(nn.Module):
    def __init__(self, in_channels, out_channels, k, stride, padding, dilation=1, groups=1, bias=True):
        super(DoGConv2D_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = int(k)
        self.kernel_size = 2*k+1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        x_coords = torch.arange(1,self.kernel_size+1,dtype=torch.float32).repeat(self.kernel_size,1)
        y_coords = torch.arange(1,self.kernel_size+1,dtype=torch.float32).view(-1,1).repeat(1, self.kernel_size)
        coords = torch.concat([x_coords.unsqueeze(0), y_coords.unsqueeze(0)], dim=0)
        kernel_dists = torch.sum(torch.square((coords-(k+1))),dim=0)
        self.kernel_dists = nn.Parameter(kernel_dists, requires_grad=False)
        self.sigma1 = nn.Parameter(torch.rand((out_channels*in_channels, 1, 1))+(k/2), requires_grad=True)
        self.sigma2_scale = nn.Parameter(torch.rand((out_channels*in_channels, 1, 1))+(k/2), requires_grad=True)
        self.total_scale = nn.Parameter(torch.randn((out_channels*in_channels, 1, 1))*2., requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=False)

    def forward(self, x):
        sigma1 = torch.clamp(self.sigma1, min=0.1)
        excite_component = (1/torch.pi*sigma1)*torch.exp(-1*self.kernel_dists/sigma1)
        sigma2 = sigma1 * torch.clamp(self.sigma2_scale, min=1+1e-4)
        inhibit_component = (1/torch.pi*sigma2)*torch.exp(-1*self.kernel_dists/torch.clamp(sigma2, min=1e-6))
        kernels = (excite_component - inhibit_component)*self.total_scale
        kernels = torch.nn.functional.normalize(kernels, dim=0)
        kernels = kernels.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(x, kernels, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class DoGConv2DLayer_v2(nn.Module):
    def __init__(self, dog_channels, k, stride, padding, dilation=1, groups=1, bias=True):
        super(DoGConv2DLayer_v2, self).__init__()
        self.dog_conv = DoGConv2D_v2(dog_channels, dog_channels, k=k, stride=stride, padding=padding, \
                                     dilation=dilation, groups=groups, bias=bias)
        self.dog_channels = dog_channels
        self.non_dog_transform = nn.Identity()

    def forward(self, x):
        x_dog = self.dog_conv(x[:,:self.dog_channels,:,:])
        x_non_dog = self.non_dog_transform(x[:,self.dog_channels:,:,:])
        return torch.concat([x_dog, x_non_dog], dim=1)


class LPP49(nn.Module):
    def __init__(self, input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation=None, subbatch_size=128):
        super(LPP49, self).__init__()
        # Polar coordinate rho and theta vals for each input location
        center_h, center_w = int(input_h/2), int(input_w/2)
        x_coords = torch.arange(input_w).repeat(input_h,1) - center_w
        y_coords = center_h - torch.arange(input_h).unsqueeze(-1).repeat(1,input_w)
        distances = torch.sqrt(x_coords**2 + y_coords**2)
        angles = torch.atan2(y_coords, x_coords)
        angles[y_coords < 0] += 2*torch.pi
        self.distances = distances
        self.angles = angles
        self.radius_bins = radius_bins
        self.angle_bins = angle_bins
        self.n_radii = len(radius_bins)-1
        self.n_angles = len(angle_bins)-1
        self.edge_radius = min(center_h, center_w)

        pooling_masks = []
        for i, (min_dist, max_dist) in enumerate(zip(radius_bins, radius_bins[1:])):
            in_distance = torch.logical_and(distances >= min_dist, distances < max_dist)
            for j, (min_angle, max_angle) in enumerate(zip(angle_bins, angle_bins[1:])):
                in_angle = torch.logical_and(angles >= min_angle, angles < max_angle)
                ind_mask = torch.logical_and(in_distance, in_angle).to(torch.float32)
                pooling_masks.append(ind_mask)
        pooling_masks = torch.stack(pooling_masks)

        if interpolation:
            for mask_idx in range(0, pooling_masks.shape[0], self.n_angles):
                radius = radius_bins[mask_idx//self.n_angles]
                if radius > self.edge_radius:
                    continue
                radius_masks = pooling_masks[mask_idx:mask_idx+self.n_angles]
                nonzero_masks = radius_masks[torch.sum(radius_masks, dim=(1,2)).to(torch.bool)]
                interpolated_masks = torch.nn.functional.interpolate(nonzero_masks.permute(1,2,0), size=self.n_angles, mode=interpolation).permute(2,0,1)
                pooling_masks[mask_idx:mask_idx+self.n_angles] = interpolated_masks
        pooling_mask_counts = torch.sum(pooling_masks, dim=(1,2))
        pooling_mask_counts[pooling_mask_counts == 0] = 1
        self.register_buffer('pooling_masks', pooling_masks)
        self.register_buffer('pooling_mask_counts', pooling_mask_counts)
        self.pooling_masks = self.pooling_masks.half()
        self.pooling_mask_counts = self.pooling_mask_counts.half()
        self.interpolation = interpolation
        self.output_transform = transforms.Resize((output_h, output_w), \
                                    interpolation=transforms.InterpolationMode.BILINEAR)
        self.subbatch_size = subbatch_size


    def forward(self, x):
        n, c, h, w = x.size()
        out = []
        for i in range(0, n, self.subbatch_size):
            # Process batch in subbatches to avoid to reduce memory consumption.  It ain't pretty, but will run on consumer gpu.
            out_i = (torch.sum(torch.mul(x[i:i+self.subbatch_size].unsqueeze(2), self.pooling_masks), dim=(-1,-2)) / self.pooling_mask_counts)
            out_i = out_i.view(out_i.size(0),c,self.n_radii,self.n_angles)
            out.append(out_i)
        out = torch.cat(out)
        out = torch.nn.functional.pad(out, (0,0,1,1), mode='reflect')
        out = torch.nn.functional.pad(out, (1,1,0,0), mode='reflect')
        out = self.output_transform(out)
        return out


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        self.padding = tuple([padding]*4)
        
    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

"""
Modification of pytorch ResNet implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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

class ConvDog(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, dilation: int = 1, bias=False):
        super(ConvDog, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.dog = DoGConv2DLayer_v2(dog_channels=in_planes//4, k=2, stride=1, padding=2, bias=False)
    def forward(self, x):
        return self.dog(self.conv(x))

def conv3x3_dog(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return ConvDog(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3_local(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> LocallyConnected2d:
    """3x3 convolution with padding"""
    return LocallyConnected2d(in_planes, out_planes, output_size=16, kernel_size=3, stride=stride, padding=dilation)

def conv1x1_local(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return LocallyConnected2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BrainBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_dog(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dn = OpponentChannelInhibition(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
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

        return self.dn(out)


class BrainBottleneckLocal(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_local(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dn = OpponentChannelInhibition(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
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

        return self.dn(out)


class ResNetPVC(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        brainblock = BrainBottleneck
        brainblock2 = BrainBottleneckLocal
        self.layer1 = self._make_micro_layer([brainblock,brainblock,brainblock2], 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        img_size = 32
        out_size = 16
        input_h, input_w = img_size, img_size
        output_h, output_w = out_size, out_size
        foveal_radius = 12
        max_dist = np.sqrt((input_h/2)**2 + (input_w/2)**2)
        foveal_dists = np.arange(0, foveal_radius)
        periphery_dists = [12, 13, 14, 16, 18, 20, 23]
        radius_bins = list(foveal_dists) + list(periphery_dists)
        angle_bins = list(np.linspace(0, 2*np.pi, img_size+1-2))
        self.lpp = LPP49(input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation='linear', subbatch_size=32)

    def _make_micro_layer(
        self,
        blockset,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * blockset[0].expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * blockset[0].expansion, stride),
                norm_layer(planes * blockset[0].expansion),
            )

        layers = []
        layers.append(
            blockset[0](
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * blockset[0].expansion
        for i in range(1, blocks):
            layers.append(
                blockset[i](
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        #layers.append(OpponentChannelInhibition(self.inplanes))

        return nn.Sequential(*layers)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        decoding_target = x
        clf_x = self.conv1(x)
        clf_x = self.bn1(clf_x)
        clf_x = self.relu(clf_x)
        
        clf_x = self.lpp(clf_x)

        clf_x = self.layer1(clf_x)
        clf_x = self.layer2(clf_x)
        clf_x = self.layer3(clf_x)
        clf_x = self.layer4(clf_x)

        clf_x = self.avgpool(clf_x)
        clf_x = torch.flatten(clf_x, 1)
        clf_x = self.fc(clf_x)

        return clf_x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet_pvc(**kwargs):
    return ResNetPVC(Bottleneck, [3, 4, 6, 3], **kwargs)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ["resnet50_primary_visual_cortex"]


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == "resnet50_primary_visual_cortex"

    identifier_short = "resnet50_primary_visual_cortex"
    url = f"https://brainscore-storage.s3.us-east-2.amazonaws.com/brainscore-vision/models/resnet50_primary_visual_cortex/{identifier_short}.pt"
    fh = urlretrieve(url, f"{identifier_short}.pth")
    load_path = fh[0]
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)

    model = resnet_pvc()
    model.load_state_dict(checkpoint)
    model.eval()

    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=64,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5)
    )

    activations_model = PytorchWrapper(
        identifier=name, 
        model=model, 
        preprocessing=preprocessing
    )

    model = ModelCommitment(
        identifier=name, 
        activations_model=activations_model,
        # specify layers to consider
        layers=['layer1']
    )

    wrapper = activations_model
    wrapper.image_size = 64
    wrapper.normalize_mean=(0.5, 0.5, 0.5)
    wrapper.normalize_std=(0.5, 0.5, 0.5)
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == "resnet50_primary_visual_cortex"

    # returns the layers you want to consider
    return  ['layer1']

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    bibtex = """
        @inproceedings{NEURIPS2023_2d1ef4ab,
            author = {Pogoncheff, Galen and Granley, Jacob and Beyeler, Michael},
            booktitle = {Advances in Neural Information Processing Systems},
            editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
            pages = {13908--13930},
            publisher = {Curran Associates, Inc.},
            title = {Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture},
            url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/2d1ef4aba0503226330661d74fdb236e-Paper-Conference.pdf},
            volume = {36},
            year = {2023}
        }
    """
    return

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
