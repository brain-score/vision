# Backends based on https://github.com/chung-neuroai-lab/adversarial-manifolds
# Removed Backends that are not used in Feather et al. 2022
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

from .modules import FakeReLU, SequentialWithArgs, FakeReLUM

# AlexNet Back-End architecture
# Based on Torchvision implementation in
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNetBackEnd(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        featurenames = [
                        'conv1', 'relu1', 'maxpool1',
                        'conv2', 'relu2',
                        'conv3', 'relu3',
                        'conv4', 'relu4',
                        'maxpool2']
        self.featurenames = featurenames

        self.fake_relu_dict = nn.ModuleDict()
        for layer_name in self.featurenames:
            if 'relu' in layer_name:
                self.fake_relu_dict[layer_name] =  FakeReLUM()

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )
        self.classifier_names = ['dropout0', 'fc0', 'fc0_relu',
                                 'dropout1', 'fc1', 'fc1_relu',
                                 'fctop']
        self.fake_relu_dict['fc0_relu'] = FakeReLUM()
        self.fake_relu_dict['fc1_relu'] = FakeReLUM()

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        all_outputs = {}
        all_outputs['input_to_backbone'] = x

        for layer, name in list(zip(self.features, self.featurenames)):
            if ('relu' in name) and fake_relu and with_latent:
                all_outputs[name + '_fake_relu'] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        x = self.avgpool(x)
        all_outputs['avgpool'] = x

        x = torch.flatten(x, 1)

        for layer, name in list(zip(self.classifier, self.classifier_names)):
            if ('relu' in name) and fake_relu and with_latent:
                all_outputs[name + '_fake_relu'] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        all_outputs['final'] = all_outputs['fctop']

        if with_latent:
            return x, None, all_outputs
        return x


