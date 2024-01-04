"""
VOneNet code based on `https://github.com/dicarlolab/vonenet` 
Only the model backbones included in Feather et al. 2022 are included. 
"""

import torch
import torch.nn as nn
import os

from .vonenet import VOneNet
from torch.nn import Module


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_vonenet_model(model_arch=None, pretrained=True, map_location='cpu', weightsdir_path=None, 
                      num_stochastic_copies=None, vone_outside_sequential=False, **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 3 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """
    print('vone_outside_sequential', vone_outside_sequential)
    if pretrained:
        if weightsdir_path is None:
            raise ValueError('The path to the saved pytorch checkpoint is not specified')

        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        if model_arch.lower() == 'resnet50_at':
            model_id = 'resnet50'
        else:
            model_id = model_arch

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                                      num_stochastic_copies=num_stochastic_copies,
                                      vone_outside_sequential=vone_outside_sequential)

        try:
            # usually this is successful, but some versions have an extra couple keys hanging out..
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module
        except:
            # if it fails, pop the problem keys and try again.
            ckpt_data['state_dict'].pop('module.vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('module.vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

        model = nn.DataParallel(model)
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch, 
                                      vone_outside_sequential=vone_outside_sequential,
                                      **kwargs)
#         nn.DataParallel(model)
    return model


def vonealexnet_gaussian_noise_std4(pretrained=False, **kwargs):
    """Constructs a Gaussian Noise VOneNet model with and AlexNet backbone

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = get_vonenet_model(model_arch='alexnet',
                              pretrained=pretrained,
                              noise_mode='gaussian',
                              noise_scale=1,
                              noise_level=4)
    return model

def vonealexnet_gaussian_fixed_noise_std4(pretrained=False, **kwargs):
    """Constructs a Gaussian Noise VOneNet model with and AlexNet backbone

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = get_vonenet_model(model_arch='alexnet',
                              pretrained=pretrained,
                              noise_mode='gaussian',
                              noise_scale=1,
                              noise_level=4)
    model.vone_block.fix_noise()
    return model


