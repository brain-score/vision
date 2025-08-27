
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
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_bibtex(model_identifier):
    return 'VGG16'

def get_model_list():
    return ['resnet50_textures_iteration=4']

def get_model(name):
    keyword = 'textures'
    iteration = 4
    network = 'resnet50'
    url = 'https://eggerbernhard.ch/shreya/latest_resnet50/textures_4.ckpt'
    output = 'resnet50_textures_iteration=4.ckpt'
    gdown.download(url, output)


    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f"resnet50_textures_iteration=4.ckpt"]
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
    assert name == 'resnet50_textures_iteration=4'
    url = 'https://eggerbernhard.ch/shreya/latest_resnet50/textures_4.ckpt'
    output = 'resnet50_textures_iteration=4.ckpt'
    gdown.download(url, output)
    layers = []
    for name, module in model._modules.items():
        print(name, "->", module)
        layers.append(name)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)

    return activations_model

def get_layers(name):
    keyword = 'textures'
    iteration = 4
    network = 'resnet50'
    url = 'https://eggerbernhard.ch/shreya/latest_resnet50/textures_4.ckpt'
    output = 'resnet50_textures_iteration=4.ckpt'
    gdown.download(url, output)


    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f"resnet50_textures_iteration=4.ckpt"]
        if len(lx_whole) > 1:
            lx_whole = [lx_whole[-1]]
    elif keyword == 'imagenet_trained' or keyword == 'no_training':
        print('keyword is imagenet')
        lx_whole = ['x']


    for model_ckpt in lx_whole:
        print(model_ckpt)
        last_module_name = None
        last_module = None
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
    layers = []
    for name, module in model._modules.items():
            print(name, "->", module)
            layers.append(name)
    return layers

if __name__ == '__main__':
    device = "cpu"
    global model
    global keyword
    global network
    global iteration
    keyword = 'textures'
    iteration = 4
    network = 'resnet50'
    url = 'https://eggerbernhard.ch/shreya/latest_resnet50/textures_4.ckpt'
    output = 'resnet50_textures_iteration=4.ckpt'
    gdown.download(url, output)


    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f"resnet50_textures_iteration=4.ckpt"]
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
    check_models.check_base_models(__name__)
