from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
from torch import nn
import pytorch_lightning as pl
import torchvision.models as models
import gdown
import glob
import os

device = "cpu"
keyword = 'less_variation'
iteration = 1
network = 'resnet50'
url = f'https://eggerbernhard.ch/shreya/latest_resnet50/less_variation_1.ckpt'
output = f'resnet50_less_variation_iteration=1.ckpt'
gdown.download(url, output)

if keyword != 'imagenet_trained' and keyword != 'no_training':
    lx_whole = list(f"resnet50_less_variation_iteration=1.ckpt")
    if len(lx_whole) > 1:
        lx_whole = [lx_whole[-1]]
elif keyword == 'imagenet_trained' or keyword == 'no_training':
    print('keyword is imagenet')
    lx_whole = ['x']

for model_ckpt in lx_whole:
    print(model_ckpt)
    last_module_name = None
    last_module = None
    layers = []
    if keyword == 'imagenet_trained' and network != 'clip':
        model = torch.hub.load('pytorch/vision', network, pretrained=True)
        for name, module in model.named_modules():
            last_module_name = name
            last_module = module
            layers.append(name)
    else:
        model = torch.hub.load('pytorch/vision', network, pretrained=False)
    if model_ckpt != 'x':
        ckpt = torch.load(model_ckpt, map_location='cpu')
    if model_ckpt != 'x' and network == 'alexnet' and keyword != 'imagenet_trained':
        ckpt2 = {}
        for keys in ckpt['state_dict']:
            print(keys)
            print(ckpt['state_dict'][keys].shape)
            print('---')
            k2 = keys.split('model.')[1]
            ckpt2[k2] = ckpt['state_dict'][keys]
        model.load_state_dict(ckpt2)
    if model_ckpt != 'x' and network == 'vgg16' and keyword != 'imagenet_trained':
        ckpt2 = {}
        for keys in ckpt['state_dict']:
            print(keys)
            print(ckpt['state_dict'][keys].shape)
            print('---')
            k2 = keys.split('model.')[1]
            ckpt2[k2] = ckpt['state_dict'][keys]
        model.load_state_dict(ckpt2)
    # Add more cases for other networks as needed

def get_bibtex(model_identifier):
    return 'VGG16'

def get_model_list():
    return [f'resnet50_less_variation_iteration=1']

def get_model(name):
    assert name == f'resnet50_less_variation_iteration=1'
    url = f'https://eggerbernhard.ch/shreya/latest_resnet50/less_variation_1.ckpt'
    output = f'resnet50_less_variation_iteration=1.ckpt'
    gdown.download(url, output)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)

    return activations_model

def get_layers(name):
    assert name == f'resnet50_less_variation_iteration=1.ckpt'
    layers = []
    url = f'https://eggerbernhard.ch/shreya/latest_resnet50/less_variation_1.ckpt'
    output = f'https://eggerbernhard.ch/shreya/latest_resnet50/less_variation_1.ckpt'
    gdown.download(url, output)
    for name, module in model.named_modules():
        layers.append(name)
    return layers

if __name__ == '__main__':
    check_models.check_base_models(__name__)
