import os
import functools

from importlib import import_module

import torch
import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

from test import test_models

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    models = []
    #models += ['vgg-11-pt']
    #models += ['alexnet', 'vgg-11-pt', 'vgg-11-bn-pt', 'vgg-13-pt', 'vgg-13-bn-pt', 'vgg-16-pt', 'vgg-16-bn-pt', 'vgg-19-pt', 'vgg-19-bn-pt']
    #models += ['squeezenet1_0', 'squeezenet1_1', 'resnet-18-pt', 'resnet-34-pt', 'resnet-50-pt', 'resnet-101-pt', 'resnet-152-pt']
    #models += ['densenet-121-pt', 'densenet-169-pt', 'densenet-201-pt', 'densenet-161-pt']
    #models += ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'resnext-50-32x4d-pt', 'resnext-101-32x8d-pt', 'mnasnet0_5-pt', 'mnasnet1_0-pt']
    models += ['resnet-50-ANT3x3_SIN']
    models += ['resnet-34-pt', 'resnet-101-pt', 'resnet-152-pt', 'resnext-50-32x4d-pt', 'resnext-101-32x8d-pt', 'mnasnet0_5-pt', 'mnasnet1_0-pt', 'resnet-50-ANT3x3_SIN']

    return models

def pytorch_model(function, image_size):
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=function, model=model_ctr(pretrained=True), preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def GoN_model(function, train, image_size):
    from urllib import request
    import torch 
    from model_tools.activations.pytorch import load_preprocess_images
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    model = model_ctr()
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    # load weights
    framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
    weightsdir_path = os.getenv('CM_TSLIM_WEIGHTS_DIR',
                                os.path.join(framework_home, 'model-weights', 'resnet-50-robust'))
    weights_id = 'resnet-50-' + train
    weights_path = os.path.join(weightsdir_path, weights_id)
    if not os.path.isfile(weights_path):
        weight_urls = { 
            'resnet-50-GNTsig0.5': 'https://github.com/bethgelab/game-of-noise/releases/download/v1.0/Gauss_sigma_0.5_Model.pth',
            'resnet-50-ANT3x3_SIN': 'https://github.com/bethgelab/game-of-noise/releases/download/v1.0/ANT3x3_SIN_Model.pth',
        }
        assert weights_id in weight_urls
        url = weight_urls[weights_id]
        os.makedirs(weightsdir_path, exist_ok=True)
        request.urlretrieve(url, weights_path)

    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    # process weights -- remove the attacker and prepocessing weights
    model.load_state_dict(checkpoint['model_state_dict'])
    # wrap model with pytorch wrapper
    wrapper = PytorchWrapper(identifier=function, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """

    image_size = 224
    if 'ANT3x3' in name:
        wrapper = GoN_model('resnet50', train='ANT3x3_SIN', image_size=224)
    else:
        wrapper = pytorch_model(model_map[name], image_size)

    return wrapper

def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    # return the same layers for all models, as they are all resnet50s
    return layer_map[name] 

model_map = {
    'alexnet':'alexnet', 
    'vgg-11-pt': 'vgg11',
    'vgg-11-bn-pt' : 'vgg11_bn', 
    'vgg-13-pt':'vgg13', 
    'vgg-13-bn-pt':'vgg13_bn', 
    'vgg-16-pt':'vgg16', 
    'vgg-16-bn-pt':'vgg16_bn', 
    'vgg-19-pt':'vgg19', 
    'vgg-19-bn-pt':'vgg19_bn',
    'squeezenet1_0':'squeezenet1_0', 
    'squeezenet1_1':'squeezenet1_1', 
    'resnet-18-pt':'resnet18', 
    'resnet-34-pt':'resnet34', 
    'resnet-50-pt':'resnet50', 
    'resnet-101-pt':'resnet101', 
    'resnet-152-pt':'resnet152',
    'densenet-121-pt':'densenet121', 
    'densenet-169-pt':'densenet169', 
    'densenet-201-pt':'densenet201', 
    'densenet-161-pt':'densenet161',
    'shufflenet_v2_x0_5':'shufflenet_v2_x0_5', 
    'shufflenet_v2_x1_0':'shufflenet_v2_x1_0',
    'resnext-50-32x4d-pt':'resnext50_32x4d', 
    'resnext-101-32x8d-pt':'resnext101_32x8d', 
    'mnasnet0_5-pt':'mnasnet0_5', 
    'mnasnet1_0-pt':'mnasnet1_0'
}

layer_map = {
    'alexnet':
        [  # conv-relu-[pool]{1,2,3,4,5}
            'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
            'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
    'vgg-11-pt': [f'features.{i}' for i in [2,5,10,15,20]] + ['classifier.1', 'classifier.4'],
    'vgg-11-bn-pt': [f'features.{i}' for i in [3,7,14,21,28]] + ['classifier.1', 'classifier.4'],
    'vgg-13-pt': [f'features.{i}' for i in [4,9,14,19,24]] + ['classifier.1', 'classifier.4'],
    'vgg-13-bn-pt': [f'features.{i}' for i in [6,13,20,27,34]] + ['classifier.1', 'classifier.4'],
    'vgg-16-pt': [f'features.{i}' for i in [4,9,16,23,30]] + ['classifier.1', 'classifier.4'],
    'vgg-16-bn-pt': [f'features.{i}' for i in [6,13,23,33,43]] + ['classifier.1', 'classifier.4'],
    'vgg-19-pt': [f'features.{i}' for i in [4,9,18,27,36]] + ['classifier.1', 'classifier.4'],
    'vgg-19-bn-pt': [f'features.{i}' for i in [6,13,26,39,52]] + ['classifier.1', 'classifier.4'],
    'squeezenet1_0':
        ['features.' + layer for layer in
         ['2'] + [f'{i}.expand3x3_activation' for i in [3, 4, 5, 7, 8, 9, 10, 12]]
         ],
    'squeezenet1_1':
        ['features.' + layer for layer in
         ['2'] + [f'{i}.expand3x3_activation' for i in [3, 4, 6, 7, 9, 10, 11, 12]]
         ],
    'densenet-121':
        ['conv1/relu'] + ['pool1'] +
        [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
        [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
        [f'conv4_block{i + 1}_concat' for i in range(24)] + ['pool4_pool'] +
        [f'conv5_block{i + 1}_concat' for i in range(16)] + ['avg_pool'],
    'densenet-169':
        ['conv1/relu'] + ['pool1'] +
        [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
        [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
        [f'conv4_block{i + 1}_concat' for i in range(32)] + ['pool4_pool'] +
        [f'conv5_block{i + 1}_concat' for i in range(32)] + ['avg_pool'],
    'densenet-201':
        ['conv1/relu'] + ['pool1'] +
        [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
        [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
        [f'conv4_block{i + 1}_concat' for i in range(48)] + ['pool4_pool'] +
        [f'conv5_block{i + 1}_concat' for i in range(32)] + ['avg_pool'],
    'xception':
        [f'block1_conv{i + 1}_act' for i in range(2)] +
        ['block2_sepconv2_act'] +
        [f'block3_sepconv{i + 1}_act' for i in range(2)] +
        [f'block4_sepconv{i + 1}_act' for i in range(2)] +
        [f'block5_sepconv{i + 1}_act' for i in range(3)] +
        [f'block6_sepconv{i + 1}_act' for i in range(3)] +
        [f'block7_sepconv{i + 1}_act' for i in range(3)] +
        [f'block8_sepconv{i + 1}_act' for i in range(3)] +
        [f'block9_sepconv{i + 1}_act' for i in range(3)] +
        [f'block10_sepconv{i + 1}_act' for i in range(3)] +
        [f'block11_sepconv{i + 1}_act' for i in range(3)] +
        [f'block12_sepconv{i + 1}_act' for i in range(3)] +
        [f'block13_sepconv{i + 1}_act' for i in range(2)] +
        [f'block14_sepconv{i + 1}_act' for i in range(2)] +
        ['avg_pool'],
    'resnet-18-pt':
        ['relu', 'maxpool'] +
        ['layer1.0.relu', 'layer1.1.relu'] +
        ['layer2.0.relu', 'layer2.0.downsample.0', 'layer2.1.relu'] +
        ['layer3.0.relu', 'layer3.0.downsample.0', 'layer3.1.relu'] +
        ['layer4.0.relu', 'layer4.0.downsample.0', 'layer4.1.relu'] +
        ['avgpool'],
    'resnet-34-pt':
        ['relu', 'maxpool'] +
        ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu'] +
        ['layer2.0.downsample.0', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu'] +
        ['layer3.0.downsample.0', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu',
         'layer3.4.relu', 'layer3.5.relu'] +
        ['layer4.0.downsample.0', 'layer4.1.relu', 'layer4.2.relu'] +
        ['avgpool'],
    'resnet-50-pt':
        ['relu', 'maxpool'] +
        ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu'] +
        ['layer2.0.downsample.0', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu'] +
        ['layer3.0.downsample.0', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu',
         'layer3.4.relu', 'layer3.5.relu'] +
        ['layer4.0.downsample.0', 'layer4.1.relu', 'layer4.2.relu'] +
        ['avgpool'],
    'resnet-50-ANT3x3_SIN':
        ['relu', 'maxpool'] +
        ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu'] +
        ['layer2.0.downsample.0', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu'] +
        ['layer3.0.downsample.0', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu',
         'layer3.4.relu', 'layer3.5.relu'] +
        ['layer4.0.downsample.0', 'layer4.1.relu', 'layer4.2.relu'] +
        ['avgpool'],
    'resnet-101-pt':
        ['relu', 'maxpool'] +
        [f'layer1.{i}.relu' for i in range(3)] +
        [f'layer2.{i}.relu' for i in range(4)] +
        [f'layer3.{i}.relu' for i in range(23)] +
        [f'layer4.{i}.relu' for i in range(3)] +
        ['avgpool'],
    'resnet-152-pt':
        ['relu', 'maxpool'] +
        [f'layer1.{i}.relu' for i in range(3)] +
        [f'layer2.{i}.relu' for i in range(4)] +
        [f'layer3.{i}.relu' for i in range(36)] +
        [f'layer4.{i}.relu' for i in range(3)] +
        ['avgpool'],
    'resnext-50-32x4d-pt':
        ['relu', 'maxpool'] +
        ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu'] +
        ['layer2.0.downsample.0', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu'] +
        ['layer3.0.downsample.0', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu',
         'layer3.4.relu', 'layer3.5.relu'] +
        ['layer4.0.downsample.0', 'layer4.1.relu', 'layer4.2.relu'] +
        ['avgpool'],
    'resnext-101-32x8d-pt':
        ['relu', 'maxpool'] +
        [f'layer1.{i}.relu' for i in range(3)] +
        [f'layer2.{i}.relu' for i in range(4)] +
        [f'layer3.{i}.relu' for i in range(23)] +
        [f'layer4.{i}.relu' for i in range(3)] +
        ['avgpool'],
    'densenet-121-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu2' for i in range(6)] +
        ['features.transition1.pool'] +
        [f'features.denseblock2.denselayer{i + 1}.relu2' for i in range(12)] +
        ['features.transition2.pool'] +
        [f'features.denseblock3.denselayer{i + 1}.relu2' for i in range(24)] +
        ['features.transition3.pool'] +
        [f'features.denseblock4.denselayer{i + 1}.relu2' for i in range(16)],
    'densenet-169-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu2' for i in range(6)] +
        ['features.transition1.pool'] +
        [f'features.denseblock2.denselayer{i + 1}.relu2' for i in range(12)] +
        ['features.transition2.pool'] +
        [f'features.denseblock3.denselayer{i + 1}.relu2' for i in range(32)] +
        ['features.transition3.pool'] +
        [f'features.denseblock4.denselayer{i + 1}.relu2' for i in range(32)],
    'densenet-201-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu2' for i in range(6)] +
        ['features.transition1.pool'] +
        [f'features.denseblock2.denselayer{i + 1}.relu2' for i in range(12)] +
        ['features.transition2.pool'] +
        [f'features.denseblock3.denselayer{i + 1}.relu2' for i in range(48)] +
        ['features.transition3.pool'] +
        [f'features.denseblock4.denselayer{i + 1}.relu2' for i in range(32)],
    'densenet-161-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu2' for i in range(6)] +
        ['features.transition1.pool'] +
        [f'features.denseblock2.denselayer{i + 1}.relu2' for i in range(12)] +
        ['features.transition2.pool'] +
        [f'features.denseblock3.denselayer{i + 1}.relu2' for i in range(36)] +
        ['features.transition3.pool'] +
        [f'features.denseblock4.denselayer{i + 1}.relu2' for i in range(24)],
    'shufflenet_v2_x0_5':
        ['maxpool'] +
        [f'stage2.{i}.branch2.7' for i in range(4)] +
        [f'stage3.{i}.branch2.7' for i in range(8)] +
        [f'stage4.{i}.branch2.7' for i in range(4)] +
        ['conv5.2'],
    'shufflenet_v2_x1_0':
        ['maxpool'] +
        [f'stage2.{i}.branch2.7' for i in range(4)] +
        [f'stage3.{i}.branch2.7' for i in range(8)] +
        [f'stage4.{i}.branch2.7' for i in range(4)] +
        ['conv5.2'],
    'mnasnet0_5-pt':
        ['layers.2', 'layers.5'] +
        [f'layers.8.{i}.layers.5' for i in range(3)] +
        [f'layers.9.{i}.layers.5' for i in range(3)] +
        [f'layers.11.{i}.layers.5' for i in range(2)] +
        [f'layers.12.{i}.layers.5' for i in range(4)] +
        [f'layers.13.{i}.layers.5' for i in range(1)] +
        ['layers.16'],
    'mnasnet1_0-pt':
        ['layers.2', 'layers.5'] +
        [f'layers.8.{i}.layers.5' for i in range(3)] +
        [f'layers.9.{i}.layers.5' for i in range(3)] +
        [f'layers.11.{i}.layers.5' for i in range(2)] +
        [f'layers.12.{i}.layers.5' for i in range(4)] +
        [f'layers.13.{i}.layers.5' for i in range(1)] +
        ['layers.16'],
}


if __name__ == '__main__':
    test_models.test_base_models(__name__)
