# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
#from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
#from candidate_models import s3 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math
import os
from torch.utils import model_zoo
#from torchvision import datasets, models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


   
import logging
import sys

"""
Template module for a base model submission to brain-score
"""

import torchvision 
import math
model_ft = torchvision.models.resnet50(pretrained=False)
#model = torch.nn.DataParallel(model).cuda()
ckpt = model_zoo.load_url("https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar"
    , map_location = device)

state_dict = ckpt['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    new_state_dict[k] = v
state_dict = new_state_dict

preprocessing = functools.partial(load_preprocess_images, image_size=224)
#dir_path = os.path.dirname(os.path.realpath(""))
#download_file("ResNet50_Manifold_mixup.pth", "ResNet50_Manifold_mixup.pth")
#model_ft = models.resnet50(pretrained=False)
#model_ft = ResNet('imagenet', depth = 50, num_classes= 1000 )

model_ft.load_state_dict(state_dict)


#model_ft = skgrcnn55() #models.resnet50(pretrained=True)
#model_ft.load_state_dict(torch.load("checkpoint_params_grcnn55_weight_share.pt"))

all_layers = [layer for layer, _ in model_ft.named_modules()]
all_layers = all_layers[1:]
model_ft = model_ft.to(device)


# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='resnet_SIN_IN_FT_IN', model= model_ft , preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='resnet_SIN_IN_FT_IN', activations_model=activations_model,
                        # specify layers to consider
                        layers=all_layers)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['resnet_SIN_IN_FT_IN']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet_SIN_IN_FT_IN'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
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

    # quick check to make sure the model is the correct one:
    assert name == 'resnet_SIN_IN_FT_IN'

    # returns the layers you want to consider
    return  all_layers

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)