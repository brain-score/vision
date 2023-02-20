"""
Read PyTorch model from .pth.tar checkpoint.
Code from:
https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py
Accessed 02.03.2020
"""

import os

import torch
from torch.utils import model_zoo
from .resnet import resnet50
from .vgg import vgg16
from .alexnet import alexnet

__all__ = ['texture_shape_alexnet_trained_on_SIN', 'texture_shape_resnet50_trained_on_SIN', 'texture_shape_vgg16_trained_on_SIN']

model_urls = {
    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    'vgg16_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar',
    'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
}

def remap_checkpoint_keys(state_dict):
    old_keys = list(state_dict.keys())
    for key in old_keys:
        if 'module.' in key:
            new_key = key.replace('module.', '')
            print(new_key)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict
        

def load_model(model_name):
    if "resnet50" in model_name:
        model = resnet50(pretrained=False)
#         model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=torch.device('cpu'))
    elif "vgg16" in model_name:
        # download model from URL manually and save to desired location
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

        assert os.path.exists(
            filepath), "Please download the VGG model yourself from the following link and save it locally: " \
                       "https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be " \
                       "downloaded automatically like the other models) "

        model = vgg16(pretrained=False)
#         model.features = torch.nn.DataParallel(model.features)
#         model.cuda()
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    elif "alexnet" in model_name:
        model = alexnet(pretrained=False)
#         model.features = torch.nn.DataParallel(model.features)
#         model.cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=torch.device('cpu'))
    else:
        raise ValueError("unknown model architecture.")

    state_dict = remap_checkpoint_keys(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    return model

def texture_shape_alexnet_trained_on_SIN(**kwargs):
    model = load_model('alexnet_trained_on_SIN')
    return model

def texture_shape_resnet50_trained_on_SIN(**kwargs):
    model = load_model('resnet50_trained_on_SIN')
    return model

def texture_shape_vgg16_trained_on_SIN(**kwargs):
    model = load_model('vgg16_trained_on_SIN')
    return model
