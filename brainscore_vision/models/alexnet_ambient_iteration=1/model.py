
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
import gdown
from torch import nn
import torchvision.models as models


def get_bibtex(model_identifier):
    return ""

def get_model_list(network, keyword, iteration):
    return [f"alexnet_ambient_iteration=1"]

def get_layers(name):
    keyword = f'ambient'
    iteration = f'1'
    network = f'alexnet'
    url = 'https://eggerbernhard.ch/shreya/latest_alexnet/ambient_1.ckpt'
    output = f'alexnet_ambient_iteration=1.ckpt'
    gdown.download(url, output)


    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f"resnet50_z_axis_iteration=3.ckpt"]
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

def get_model(name):
    keyword = f'ambient'
    iteration = f'1
    network = f'alexnet'
    url = 'https://eggerbernhard.ch/shreya/latest_alexnet/ambient_1.ckpt'
    output = f'alexnet_ambient_iteration=1.ckpt'
    gdown.download(url, output)


    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f'alexnet_ambient_iteration=1.ckpt']
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
    assert name == 'alexnet_ambient_iteration=1'
    url = 'https://eggerbernhard.ch/shreya/latest_alexnet/ambient_1.ckpt'
    output = f'alexnet_ambient_iteration=1.ckpt'
    gdown.download(url, output)
    layers = []
    for name, module in model._modules.items():
        print(name, "->", module)
        layers.append(name)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)

    return activations_model

if __name__ == '__main__':
    device = "cpu"
    network = f"alexnet"  # Example network
    keyword = f"ambient"  # Example keyword
    iteration = f"1"  # Example iteration

    url = f"https://eggerbernhard.ch/shreya/latest_alexnet/ambient_1.ckpt"
    output = f"alexnet_ambient_iteration=1.ckpt"
    gdown.download(url, output)

    model = get_model(network, keyword, iteration)
    layers = get_layers(network, keyword, iteration)
    print(f"Loaded model:", model)
    print(f"Available layers:", layers)
