from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import os
import ssl
from brainscore_core.supported_data_standards.brainio.s3 import load_file


ssl._create_default_https_context = ssl._create_unverified_context

torch.set_default_dtype(torch.float64)


class SeparableConv2d(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, stride=1, padding='same'):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            num_in_channels, num_in_channels, kernel_size, stride, padding, groups=num_in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(num_in_channels, num_out_channels, 1, 1, 0, bias=True)  # 1x1 convolution

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CompactModel(nn.Module):
    def __init__(self):
        super(CompactModel, self).__init__()
        nums_filters = [50,50,50,100,100]
        num_neurons = 88

        self.layer0_conv = nn.Conv2d(3, nums_filters[0], kernel_size=(5,5), stride=1, padding='same', bias=True)  # Convolutional layer
        self.layer0_bn = nn.BatchNorm2d(num_features=nums_filters[0])
        self.layer0_act = nn.ReLU()
        self.layer1_conv_depth = nn.Conv2d(nums_filters[0], nums_filters[0], kernel_size=(5,5), stride=2, padding=0, groups=nums_filters[0], bias=False)
        self.layer1_conv_point = nn.Conv2d(nums_filters[0], nums_filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.layer1_bn = nn.BatchNorm2d(num_features=nums_filters[1])
        self.layer1_act = nn.ReLU()
        self.layer2_conv_depth = nn.Conv2d(nums_filters[1], nums_filters[1], kernel_size=(5,5), stride=2, padding=0, groups=nums_filters[1], bias=False)
        self.layer2_conv_point = nn.Conv2d(nums_filters[1], nums_filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.layer2_bn = nn.BatchNorm2d(num_features=nums_filters[2])
        self.layer2_act = nn.ReLU()
        self.layer3_conv_depth = nn.Conv2d(nums_filters[2], nums_filters[2], kernel_size=5, stride=1, padding='same', groups=nums_filters[2], bias=False)
        self.layer3_conv_point = nn.Conv2d(nums_filters[2], nums_filters[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.layer3_bn = nn.BatchNorm2d(num_features=nums_filters[3])
        self.layer3_act = nn.ReLU()
        self.layer4_conv_depth = nn.Conv2d(nums_filters[3], nums_filters[3], kernel_size=5, stride=1, padding='same', groups=nums_filters[3], bias=False)
        self.layer4_conv_point = nn.Conv2d(nums_filters[3], nums_filters[4], kernel_size=1, stride=1, padding=0, bias=True)
        self.layer4_bn = nn.BatchNorm2d(num_features=nums_filters[4])
        self.layer4_act = nn.ReLU()
        self.mixing_stage = nn.Conv2d(nums_filters[4], num_neurons, kernel_size=(1,1), stride=1, padding='same')
        self.spatial_pool_stage = nn.Conv2d(num_neurons, num_neurons, kernel_size=(28,28), stride=1, groups=num_neurons)


    def forward(self, x):

        # preprocessing (and reversing some preprocessing from BrainScore to fit compact model preprocessing)
        x = x[:, :, ::2, ::2]

        means_normalize = [0.485, 0.456, 0.406]
        stds_normalize = [0.229, 0.224, 0.225]
        x[:,0,:,:] = stds_normalize[0] * x[:,0,:,:] + means_normalize[0]
        x[:,1,:,:] = stds_normalize[1] * x[:,1,:,:] + means_normalize[1]
        x[:,2,:,:] = stds_normalize[2] * x[:,2,:,:] + means_normalize[2]
        x = 256. * x  # convert to pixel intensities uint8 (0 to 255)

        rgb_vals = [116.222, 109.270, 100.381]
        for ichannel in range(3):  # recenter
            x[:,ichannel,:,:] = x[:,ichannel,:,:] - rgb_vals[-ichannel]  # pytorch in bgr

        x = self.layer0_act(self.layer0_bn(self.layer0_conv(x)))
        x = F.pad(x, (1,3,1,3))
        x = self.layer1_act(self.layer1_bn(self.layer1_conv_point(self.layer1_conv_depth(x))))
        x = F.pad(x, (1,3,1,3))
        x = self.layer2_act(self.layer2_bn(self.layer2_conv_point(self.layer2_conv_depth(x))))
        x = self.layer3_act(self.layer3_bn(self.layer3_conv_point(self.layer3_conv_depth(x))))
        x = self.layer4_act(self.layer4_bn(self.layer4_conv_point(self.layer4_conv_depth(x))))
        x = self.spatial_pool_stage(self.mixing_stage(x))

        return x


def get_model_list():
    return ['compact_resnet50_robust_V4']


def get_model(name):
    assert name == 'compact_resnet50_robust_V4'

    pytorch_device = torch.device('cpu')


    # ## access local weights
    # load_path = '../convert_shared_compact_models_to_torch/results/models_torch/compact_ResNet50_V4_50filters.pt'
    # state_dict = torch.load(load_path, map_location=pytorch_device)

    ## online via brainscore to load weights (save weights as personal file in brainscore)
    ## https://brainscore-storage.s3.us-east-2.amazonaws.com/brainscore-vision/models/user_718/compact_ResNet50_V4_50filters.pt?versionId=oa0hxVd5Yak8BhbJBEiZdYmzOm1jlbnM
    file_path = load_file(bucket="brainscore-storage", folder_name="brainscore-vision/models/user_718/",
                          relative_path="compact_ResNet50_robust_V4_50filters.pt",
                          version_id="HR8jQoPvvZpUOjDGCNU_FKDaBl_8kjtF")
    state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)  # map onto cpu

    pytorch_model = CompactModel()
    pytorch_model = pytorch_model.to(pytorch_device)

    pytorch_model.load_state_dict(state_dict, strict=True)
    pytorch_model.eval()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    wrapper = PytorchWrapper(identifier=name,
                             model=pytorch_model,
                             preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'compact_resnet50_robust_V4'
    return ['layer1_act', 'layer2_act', 'layer3_act', 'layer4_act', 'spatial_pool_stage']


def get_bibtex(model_identifier):
    return """@article{cowley2023compact,
    title={Compact deep neural network models of visual cortex},
    author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
    journal={bioRxiv},
    pages={2023--11},
    year={2023},
    publisher={Cold Spring Harbor Laboratory}
    }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)



