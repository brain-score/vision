# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
from .network_blocks import Basic2DBlock

__all__ = [
    'Conv2D'
]

class Conv2D(nn.Module):
    def __init__(self, depth=10):
        super(Conv2D, self).__init__()
        assert depth==10

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = Basic2DBlock(64, 64, stride=(2, 2))
        self.block2 = Basic2DBlock(64, 128, stride=(2, 2))
        self.block3 = Basic2DBlock(128, 256, stride=(2, 2))
        self.block4 = Basic2DBlock(256, 512)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)
        x_pool = self.pool(x_b4)
        if return_embs:
            return {'conv2x': x_b1, 'conv3x': x_b2, 'conv4x': x_b3, 'conv5x': x_b4, 'pool': x_pool}
        else:
            return x_pool
