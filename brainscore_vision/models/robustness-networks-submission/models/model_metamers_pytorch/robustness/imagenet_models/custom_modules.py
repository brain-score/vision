import torch 
from torch import nn
import numpy as np
ch = torch

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)


class HannPooling2d(nn.Module):
    """
    2D Weighted average pooling with a Hann window.

    Inputs
    ------
    stride (int): amount of subsampling
    pool_size (int or list): width of the hann window (if int, assume window is square)
    padding (string): type of padding for the convolution
    normalize (bool): if true, divide the filter by the sum of its values, so that the
        smoothed signal is the same amplitude as the original.
    """
    def __init__(self, stride, pool_size, padding=0, normalize=True):
        super(HannPooling2d, self).__init__()
        self.stride = stride
        if type(pool_size)==int:
            pool_size = [pool_size, pool_size]
        self.pool_size = pool_size
        if padding=='same':
            # Subtract 2 because we only take the non-zero parts of the 
            # hann window in _make_hann_window
            padding = [int(np.floor((p-2)/2)) for p in self.pool_size]
        self.padding = padding
        self.normalize = normalize

        hann_window2d = self._make_hann_window()
        self.register_buffer('hann_window2d', ch.from_numpy(hann_window2d).float())

    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)==2: # Assume no batch or channel dimension
            h,w = x_shape
            x = x.view(1,1,h,w)
        elif len(x_shape)==3: # Assume no batch dimension
            c,h,w = x_shape
            x = x.view(c, 1, h, w)
        elif len(x_shape)==4:
            b,c,h,w = x_shape
            x = x.view(b*c, 1, h, w)
        x = ch.nn.functional.conv2d(x, self.hann_window2d,
                                    stride=self.stride,
                                    padding=self.padding)
        x_shape_after_filt = x.shape
        return x.view(x_shape[0:-2] + (x_shape_after_filt[-2],x_shape_after_filt[-1]))

    def _make_hann_window(self):
        if self.pool_size[0]>2:
            # Remove the zeros at the edges of the hann window
            hann_window_h = np.expand_dims(np.hanning(self.pool_size[0])[1:-1],0)
        else:
            hann_window_h = np.expand_dims(np.hanning(self.pool_size[0]),0)

        if self.pool_size[1]>2:
            # Remove the zeros at the edges of the hann window
            hann_window_w = np.expand_dims(np.hanning(self.pool_size[1])[1:-1],1)
        else:
            hann_window_w = np.expand_dims(np.hanning(self.pool_size[1]),1)

        # Add a channel dimensiom to the filter
        hann_window2d = np.expand_dims(np.expand_dims(np.outer(hann_window_h, hann_window_w),0),0)

        if self.normalize:
            hann_window2d = hann_window2d/(sum(hann_window2d.ravel()))
        return hann_window2d


def ConvHPool2d(in_channels, out_channels, kernel_size, pool_size, stride=1,
                padding=1, normalize=True, **convkwargs):
    """Convolution with stride of 1 followed by weighted average pooling with hann window"""
    return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                     padding=int(np.floor(kernel_size/2)), bias=True),
                nn.ReLU(inplace=True),
                HannPooling2d(stride, pool_size, padding=padding),
            )

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input
