import torch
import os
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from timm.models.layers import Conv2dSame



dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = [
    "resnet50"
]


class DSConv2d(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin,
                                   padding_mode='reflect')
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SpatialAttn(nn.Module):
    def __init__(self, inp_dim, hidden_sizes, num_heads, atttype='channel'):
        super(SpatialAttn, self).__init__()
        self.inp_dim = inp_dim
        self.hidden_sizes = hidden_sizes
        self.num_heads = num_heads

        self.module_list = nn.ModuleList([])
        if atttype == 'channel':
            first_pool = nn.Conv2d(self.inp_dim, hidden_sizes[0], kernel_size=1, bias=True)
        elif atttype == 'dw':
            first_pool = DSConv2d(self.inp_dim, 1, hidden_sizes[0])
        for i in range(len(hidden_sizes) + 1):
            if i == 0:
                self.module_list.append(first_pool)
                self.module_list.append(nn.ReLU(inplace=True))
            elif i < len(hidden_sizes):
                self.module_list.append(nn.Conv2d(hidden_sizes[i - 1], hidden_sizes[i], kernel_size=1, bias=True))
                self.module_list.append(nn.ReLU(inplace=True))
            else:
                self.module_list.append(nn.Conv2d(hidden_sizes[i - 1], num_heads, kernel_size=1, bias=False))

    def forward(self, x):
        inp = x
        for m in self.module_list:
            x = m(x)
        assert self.inp_dim % x.shape[1] == 0, "Output channels must be divisible by input number of channels"
        x = nn.Softmax(dim=-1)(x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
        if x.shape[1] != self.inp_dim:
            x = x.repeat(1, self.inp_dim // x.shape[1], 1, 1)
        w = x
        x = w * inp
        x = x.sum((-1, -2))
        return x


class BlurPool2d(nn.Module):
    def __init__(self, kernel_size, stride, blur_kernel_learnable=False):
        super(BlurPool2d, self).__init__()
        self.blur_kernel = nn.Parameter(self._get_blur_kernel(kernel_size))
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        if not blur_kernel_learnable:
            self.blur_kernel.requires_grad_(False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(-1, H, W).unsqueeze(1)
        x = F.conv2d(x, self.blur_kernel, stride=self.stride, padding=self.padding)
        H, W = x.shape[2:]
        return x.view(B, C, H, W)

    def _get_blur_kernel(self, kernel_size):
        blur_kernel_dict = {
            2: [1, 1],
            3: [1, 2, 1],
            4: [1, 3, 3, 1],
            5: [1, 4, 6, 4, 1],
            6: [1, 5, 10, 10, 5, 1],
            7: [1, 6, 15, 20, 15, 6, 1]
        }
        if kernel_size in blur_kernel_dict.keys():
            blur_kernel_1d = torch.FloatTensor(blur_kernel_dict[kernel_size]).view(-1, 1)
            blur_kernel = torch.matmul(blur_kernel_1d, blur_kernel_1d.t())
            blur_kernel.div_(blur_kernel.sum())
            return blur_kernel.unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError("invalid blur kernel size: {}".format(kernel_size))

    def __repr__(self):
        return 'BlurPool2d(kernel_size=({}, {}), stride=({}, {}), padding=({}, {}))'.format(
            self.kernel_size, self.kernel_size, self.stride,
            self.stride, self.padding, self.padding
        )


class MaxBlurPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, blur_kernel_size=3, blur_kernel_learnable=False,
                 blur_position='after'):
        super(MaxBlurPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        self.blurpool = BlurPool2d(kernel_size=blur_kernel_size, stride=stride,
                                   blur_kernel_learnable=blur_kernel_learnable)

        if blur_position == 'after':
            self.layer = [self.maxpool, self.blurpool]
        elif blur_position == 'before':
            self.layer = [self.blurpool, self.maxpool]
        else:
            raise ValueError('invalid blur postion: {}'.format(blur_position))

        self.main = nn.Sequential(self.maxpool, self.blurpool)

    def forward(self, x):
        return self.main(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2dSame:
    """3x3 convolution with padding"""
    return Conv2dSame(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2dSame:
    """1x1 convolution"""
    return Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
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


class ResNet(nn.Module):
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
            attn_pool: bool = False,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
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
        self.conv1 = Conv2dSame(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.attnpool = attn_pool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if attn_pool:
            self.avgpool = SpatialAttn(2048, [4096, 2048], 1)
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2dSame):
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
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.attnpool:
            x2 = self.avgpool2(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops": 4.089,
            "_weight_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_weight_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


# The dictionary below is internal implementation detail and will be removed in v0.15
from torchvision.models._utils import _ModelURLs

model_urls = _ModelURLs(
    {
        "resnet50": ResNet50_Weights.IMAGENET1K_V1.url,

    }
)
