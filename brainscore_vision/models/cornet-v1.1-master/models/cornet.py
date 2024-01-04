from collections import OrderedDict

import torch
from torch import nn


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class LGN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=7, stride=2,
                              padding=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ReLU(inplace=True)
        self.output = Identity()  # no input -> no output

    def forward(self, inp):
        output = self.output(self.nonlin(self.norm(self.conv(inp))))
        return output


class CORblock(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.output = Identity()  # no input -> no output

        # change the number of channels
        self.conv_input = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, bias=False)

        self.upsample_state = nn.ConvTranspose2d(out_channels, out_channels,
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=False)

        self.conv_inp_strided = nn.Conv2d(out_channels, out_channels,
                                          kernel_size=1, stride=2, bias=False)
        self.norm_inp_strided = nn.GroupNorm(32, out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels * self.scale)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.nonlin2 = nn.ReLU()

    def forward(self, inp=None, state=None, batch_size=None):
        inp = self.conv_input(inp)

        if state is None:
            # initialize state
            state = 0
            x = inp + state
        else:
            # resize state to input size and add
            x = inp + self.upsample_state(state)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        res = self.nonlin2(x)

        # accumulate residuals
        state = state + res

        # return input + sum of residuals
        inp_strided = self.norm_inp_strided(self.conv_inp_strided(inp))
        output = self.output(inp_strided + state)

        return output, state


def Decoder(in_channels=512, out_channels=1000):
    decoder = nn.Sequential(OrderedDict([
        ('avgpool', nn.AdaptiveAvgPool2d(1)),
        ('flatten', nn.Flatten()),
        ('linear', nn.Linear(in_channels, out_channels)),
        ('output', Identity())
    ]))
    return decoder


class CORnet_v1(nn.Module):

    def __init__(self, times=7, pretrained=False, map_location='cpu'):
        super().__init__()
        self.times = times

        self.LGN = LGN(3, 32)
        self.V1 = CORblock(32, 64)
        self.V2 = CORblock(64, 128)
        self.V4 = CORblock(128, 256)
        self.IT = CORblock(256, 512)
        self.decoder = Decoder(512, 1000)

        self.conn = {'V1': 'LGN',
                     'V2': 'V1',
                     'V4': 'V2',
                     'IT': 'V4'}

        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            weights = torch.load('weights/cornet-v1.1.pth',
                                 map_location=map_location)
            self.load_state_dict(weights)

    def forward(self, inp):
        lgn_output = self.LGN(inp)

        blocks = list(reversed(list(self.conn.items())))
        block_names = list(self.conn.keys())
        cut_t = self.times - len(block_names)

        outputs = {}
        states = {}
        for t in range(self.times):
            if t <= cut_t:
                comp_blocks = block_names
            else:
                comp_blocks = block_names[t - cut_t:]

            for block, input_block in blocks:
                if input_block == 'LGN':
                    output = lgn_output
                else:
                    output = outputs.get(input_block, None)
                state = states.get(block, None)

                if output is not None and block in comp_blocks:
                    b = getattr(self, block)
                    outputs[block], states[block] = b(output, state)
        out = self.decoder(outputs['IT'])
        return out
