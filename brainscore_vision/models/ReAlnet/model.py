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
from brainscore_vision.model_helpers.s3 import load_weight_file
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import json

LAYERS = ['V1', 'V2', 'V4', 'IT', 'decoder.avgpool']

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
        ('V1', nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
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
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

class Encoder(nn.Module):
    def __init__(self, realnet, n_output):
        super(Encoder, self).__init__()
        
        # CORnet
        self.realnet = realnet
        
        # fully connected layers
        self.fc_v1 = nn.Linear(200704, 128)
        self.fc_v2 = nn.Linear(100352, 128)
        self.fc_v4 = nn.Linear(50176, 128)
        self.fc_it = nn.Linear(25088, 128)
        self.fc = nn.Linear(512, n_output)
        self.activation = nn.ReLU()
        
    def forward(self, imgs):
        # forward pass through CORnet_S
        outputs = self.realnet(imgs)
        
        N = len(imgs)
        v1_outputs = self.realnet.V1(imgs)   # N * 64 * 56 * 56
        v2_outputs = self.realnet.V2(v1_outputs)   # N * 128 * 28 * 28
        v4_outputs = self.realnet.V4(v2_outputs)   # N * 256 * 14 * 14
        it_outputs = self.realnet.IT(v4_outputs)   # N * 512 * 7 * 7

        # flatten and pass through fully connected layers
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

# Change here: use 'cpu'
device = 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Construct CORnet_S
realnet = CORnet_S()
# (Optional) remove DataParallel if not needed for CPU
# realnet = torch.nn.DataParallel(realnet)

def load_config(json_file):
    # Get the directory containing this script (model.py)
    base_dir = os.path.dirname(__file__)
    
    # Construct the path to the JSON file
    json_path = os.path.join(base_dir, json_file)
    
    # Read the JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Build encoder model
encoder = Encoder(realnet, 340)
def model_load_weights(identifier: str):
    # Download weights (Brain-Score team modification)
    # Read the version id and sha1 from json file called "weights.json"
    weights_info = load_config("weights.json")

    version_id = weights_info['version_ids'][identifier]
    sha1 = weights_info['sha1s'][identifier]

    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                        relative_path=f"ReAlnet/{identifier}_best_model_params.pt",
                                        version_id=version_id,
                                        sha1=sha1)

    # Load weights onto CPU and remove "module." from keys
    weights = torch.load(weights_path, map_location='cpu')
    new_state_dict = {}
    for key, val in weights.items():
        # remove "module." (if it exists) from the key
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val

    encoder.load_state_dict(new_state_dict)

    # Retrieve the realnet portion from the encoder
    realnet = encoder.realnet
    realnet.eval()
    return realnet

def get_model(identifier: str):
    model = model_load_weights(identifier)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

# if __name__ == "__main__":