import functools

from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

from brainscore.benchmarks import public_benchmark_pool
from brainscore import score_model
from model_tools.check_submission import check_models
from model_tools.brain_transformation import ModelCommitment
import numpy as np

import socket


from skimage import draw
import copy

import torch
import torch as ch

import torchvision.models as torchmodels
from torchvision import transforms
# check you have the right version of timm
import timm
# assert timm.__version__ == "0.3.2"

import torch.nn as nn
from collections import OrderedDict

import sys
import os

sys.path.append(".")

print(os.path.dirname(os.path.realpath('.')))
print(os.path.realpath('.'))

sys.path.insert(0, './models')

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

# from resnet_joker import resnet50 as joker_resnet50

# helper funcs

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from robustness.tools.custom_modules import SequentialWithArgs, FakeReLU

new_mean_robust_madry = torch.tensor([0.4850, 0.4560, 0.4060])
new_std_robust_madry = torch.tensor([0.2290,0.2240,0.2250])

def get_model_list():
    return ['rs_sigma_0p5']

class InputNormalize(nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized
    
    
class MadryRobust(nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, normalizer, model):
        super(MadryRobust, self).__init__()
        
        self.normalizer = normalizer
        self.model = model

    def forward(self, x):
        x = self.normalizer(x)
        x = self.model(x)
        return x


def get_model(name):
    assert name == 'rs_sigma_0p5'

    model =  torchmodels.resnet50(pretrained=False) 
    if socket.gethostname() == 'turing':
        path_previous_checkpoint = './checkpoint_sigma0p50.pth.tar'
    else:
        path_previous_checkpoint = os.environ['HOME']+'/work_dir/rs_robustness_submission_sigma0p5/models/checkpoint_sigma0p50.pth.tar'


    checkpoint = torch.load(path_previous_checkpoint, map_location="cpu")['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[9:] # remove `module.`
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict)
    assert len(msg.missing_keys)==0
    assert np.all(['attacker' in m for m in msg.unexpected_keys])



    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='rs_sigma_0p5', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper




def get_layers(name):
    assert name == 'rs_sigma_0p5'
    return [
        'conv1',
        'bn1',
        'relu',
        'maxpool',
        'layer1',
        'layer1.0.conv1',
        'layer1.0.conv2',
        'layer1.0.conv3',
        'layer1.0.bn3',
        'layer1.0.relu',
        'layer1.0.downsample',
        'layer1.0.downsample.0',
        'layer1.0.downsample.1',

        'layer2.0.downsample.0',
        'layer2.0.conv1',
        'layer2.0.conv2',
        'layer2.0.conv3',
        'layer2.0.bn3',
        'layer2.0.relu',

        'layer3.0.downsample.0',
        'layer3.0.conv1',
        'layer3.0.conv2',
        'layer3.0.conv3',
        'layer3.0.bn3',
        'layer3.0.relu',


        'layer4.0.downsample.0',
        'layer4.0.conv1',
        'layer4.0.conv2',
        'layer4.0.conv3',
        'layer4.0.bn3',
        'layer4.0.relu',

        'avgpool'


    ]

def get_bibtex(model_identifier):
    return """
    """


if __name__ == '__main__':
    with torch.no_grad():
        check_models.check_base_models(__name__)
