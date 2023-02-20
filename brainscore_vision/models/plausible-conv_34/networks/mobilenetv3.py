"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
Implementation adapted from https://github.com/d-li14/mobilenetv3.pytorch
"""

import torch.nn as nn
import math
import numpy as np
from networks import network_utils, locally_connected_utils

__all__ = ['mobilenetv3_lc_large', 'mobilenetv3_lc_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, in_size, is_approx=False):
    if is_approx:
        return nn.Sequential(
            locally_connected_utils.LocallyConnected2d(in_size, inp, oup, 3, stride=stride),
            nn.BatchNorm2d(oup),
            h_swish()
        )
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup, in_size, is_approx=False):
    if is_approx:
        return nn.Sequential(
            locally_connected_utils.LocallyConnected2d(in_size, inp, oup, 1, stride=1),
            nn.BatchNorm2d(oup),
            h_swish()
        )
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, in_size,
                 is_approx=False, conv_1x1=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            if is_approx:
                conv_one = locally_connected_utils.LocallyConnected2d(
                    in_size, hidden_dim, hidden_dim, kernel_size,
                    stride=stride, groups=hidden_dim)
                if conv_1x1:
                    conv_two = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
                else:
                    conv_two = locally_connected_utils.LocallyConnected2d(
                        [in_size[0] // stride, in_size[1] // stride], hidden_dim, oup, 1, stride=1)
            else:
                conv_one = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                                     groups=hidden_dim, bias=False)
                conv_two = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)

            self.conv = nn.Sequential(
                # dw
                conv_one,
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                conv_two,
                nn.BatchNorm2d(oup),
            )
        else:
            if is_approx:
                if conv_1x1:
                    conv_one = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
                    conv_three = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
                else:
                    conv_one = locally_connected_utils.LocallyConnected2d(in_size, inp, hidden_dim, 1, stride=1)
                    conv_three = locally_connected_utils.LocallyConnected2d([in_size[0] // stride, in_size[1] // stride],
                                                                            hidden_dim, oup, 1, stride=1)

                conv_two = locally_connected_utils.LocallyConnected2d(in_size, hidden_dim, hidden_dim, kernel_size,
                                                                      stride=stride,
                                                                      groups=hidden_dim)
            else:
                conv_one = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
                conv_two = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                                     groups=hidden_dim, bias=False)
                conv_three = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)

            self.conv = nn.Sequential(
                # pw
                conv_one,
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                conv_two,
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                conv_three,
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3LC(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., in_size=np.array([224, 224]), is_approx=False,
                 first_conv=False, conv_1x1=False, wishart_1x1=False):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.wishart_1x1 = wishart_1x1
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, in_size, is_approx and not first_conv)]
        in_size = in_size // 2
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, in_size,
                                is_approx, conv_1x1))
            in_size = in_size // s
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size, in_size, is_approx and not conv_1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]

        dropout_rate = 0.1 if (width_mult < 1.0 and mode == 'small') else 0.2

        self.encoder_dim = exp_size
        self.n_classes = num_classes

        self.decoder = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()
        self.hidden_layers = self._hidden_layers
        self.n_hidden_layer = None

    def get_n_hidden_layers(self):
        # todo: fix
        if self.n_hidden_layer is None:
            self.n_hidden_layer = 0
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, locally_connected_utils.LocallyConnected2d):
                    self.n_hidden_layer += 1
        return self.n_hidden_layer

    def _hidden_layers(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        # finished_net = torch.load('logs_imagenet_short/mbv3_256_200ep_approx_first_1x1/net_final.pth').cpu()
        # saved_1x1 = []
        # with torch.no_grad():
        #     for m in finished_net.modules():
        #         if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 1:
        #             saved_1x1.append(m.weight.data)
        # ind_1x1 = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # if self.wishart_1x1 and m.kernel_size[0] == 1:
                #     # print('SOLUTION INIT 1x1')
                #     # m.weight.data = saved_1x1[ind_1x1].clone()
                #     # ind_1x1 += 1
                #     print('WISHART INIT')
                #     ch_out, ch_in = m.weight.shape[:2]
                #     cov_dim, degrees_of_freedom = min(ch_out, ch_in), max(ch_out, ch_in)
                #     basis = ortho_group.rvs(cov_dim)
                #     cov_spectrum = np.diag(np.exp(-np.abs(np.linspace(0, 1, cov_dim)) / 0.3) * 0.3) \
                #                     / degrees_of_freedom
                #     covariance_matrix = torch.tensor(wishart.rvs(df=degrees_of_freedom,
                #                                                  scale=basis.dot(cov_spectrum).dot(basis.T)),
                #                                      dtype=torch.float)
                #     distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                #         torch.zeros(cov_dim), covariance_matrix=covariance_matrix)
                #     if ch_out == degrees_of_freedom:
                #         m.weight.data = distribution.sample((degrees_of_freedom,))[:, :, None, None]
                #     else:
                #         m.weight.data = distribution.sample((degrees_of_freedom,))[:, :, None, None].transpose(0, 1)
                # else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # same as
                # gain = torch.nn.init.calculate_gain('relu')
                # weight_std = gain / np.sqrt(out_channels * kernel_size ** 2)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        return self.decoder(self.hidden_layers(x))


def mobilenetv3_lc_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3LC(cfgs, mode='large', **kwargs)


def mobilenetv3_lc_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3LC(cfgs, mode='small', **kwargs)