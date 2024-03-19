import torch.nn as nn

from torchvision.models.video.resnet import (
    BasicBlock,
    Bottleneck,
    BasicStem,
    R2Plus1dStem,
    _video_resnet,
)


__all__ = ["r2plus1d_18"]


def r2plus1d_18(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "r2plus1d_18",
        False,
        False,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    return model


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
