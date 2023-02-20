""" Conv2d w/ Same Padding
Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .padding import pad_same, get_padding_value, pad_same1d


def conv1d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same1d(x, weight.shape[-2:], stride, dilation)
    return F.conv1d(x, weight, bias, stride, 0, dilation, groups)


class Conv1dSame(nn.Conv1d):
    """ Tensorflow like 'SAME' convolution wrapper for 1D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv1d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv1d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv1dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv1d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)