""" AvgPool2d w/ Same Padding
Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .helpers import tup_pair
from .padding import pad_same, get_padding_value, pad_same1d


def avg_pool1d_same(x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
                    ceil_mode: bool = False, count_include_pad: bool = True):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same1d(x, kernel_size, stride)
    kernel_size = kernel_size[0]
    stride = stride[0]
    return F.avg_pool1d(x, kernel_size, stride, ( 0), ceil_mode, count_include_pad)


class AvgPool1dSame(nn.AvgPool1d):
    """ Tensorflow like 'SAME' wrapper for 1D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(AvgPool1dSame, self).__init__(kernel_size, stride, (0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool1d_same(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


def max_pool1d_same(
         x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
        dilation: List[int] = (1, 1), ceil_mode: bool = False):
    x = pad_same1d(x, kernel_size, stride, value=-float('inf'))
    kernel_size = kernel_size[0]
    stride = stride[0]
    return F.max_pool1d(x, kernel_size, stride, (0), dilation, ceil_mode)


class MaxPool1dSame(nn.MaxPool1d):
    """ Tensorflow like 'SAME' wrapper for 1D max pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(MaxPool1dSame, self).__init__(kernel_size, stride, (0), dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        return max_pool1d_same(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


def create_pool1d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool1dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool1dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    else:
        if pool_type == 'avg':
            return nn.AvgPool1d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == 'max':
            return nn.MaxPool1d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'