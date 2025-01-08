import math
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms
import torch.utils.model_zoo
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F
import h5py
import random

import functools

import torchvision.models

import gdown

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images



LAYERS = ['module.V1', 'module.V2', 'module.V4', 'module.IT', 'module.decoder.avgpool']

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


class Encoder(nn.Module):
    def __init__(self, realnet, n_output):
        super(Encoder, self).__init__()
        
        # CORnet
        self.realnet = realnet
        
        # full connected layer
        self.fc_v1 = nn.Linear(200704, 128)
        self.fc_v2 = nn.Linear(100352, 128)
        self.fc_v4 = nn.Linear(50176, 128)
        self.fc_it = nn.Linear(25088, 128)
        self.fc = nn.Linear(512, n_output)
        self.activation = nn.ReLU()
        
    def forward(self, imgs):
        
        outputs = self.realnet(imgs)
        
        N = len(imgs)
        v1_outputs = self.realnet.module.V1(imgs) # N * 64 * 56 * 56
        v2_outputs = self.realnet.module.V2(v1_outputs) # N * 128 * 28 * 28
        v4_outputs = self.realnet.module.V4(v2_outputs) # N * 256 * 14 * 14
        it_outputs = self.realnet.module.IT(v4_outputs) # N * 512 * 7 * 7
        v1_features = self.fc_v1(v1_outputs.view(N, -1))
        v1_features = self.activation(v1_features)
        v2_features = self.fc_v2(v2_outputs.view(N, -1))
        v2_features = self.activation(v2_features)
        v4_features = self.fc_v4(v4_outputs.view(N, -1))
        v4_features = self.activation(v4_features)
        it_features = self.fc_it(it_outputs.view(N, -1))
        it_features = self.activation(it_features)
        features = torch.cat((v1_features, v2_features, v4_features, it_features), dim=1)
        features = self.fc(features)
        
        return outputs, features


torch.set_default_dtype(torch.float32)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

device='cuda'

realnet = CORnet_S().to(device)
realnet = torch.nn.DataParallel(realnet)
encoder = Encoder(realnet, 340).to(device)
# weights = torch.load('/Users/yat-lok/Desktop/ReAInet/ReAInet_weights/sub-01/best_model_params.pt')
# weights = torch.load('/home/yilwang_umass_edu/ReAlnet_weights/sub-01/best_model_params.pt', map_location=device)
# weights = torch.load('/work/pi_gstuart_umass_edu/yile/weights_1/sub-01.pt', map_location=device)
# weights = torch.load('/work/pi_gstuart_umass_edu/yile/weights_10/sub-01.pt', map_location=device)

url = 'https://drive.google.com/uc?id=1AtpS7dPV8t3e1aT8a4Nu-mIFkbWRH-ff'
output_file = "best_model_params.pt"
gdown.download(url, output_file, quiet = False)

weights = torch.load(output_file, map_location=device)


encoder.load_state_dict(weights)
realnet = encoder.realnet

realnet.eval()

def get_model():
    model = realnet
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='ReAlnet01', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
