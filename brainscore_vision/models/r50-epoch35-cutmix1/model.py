# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
import torch.nn.functional as F
import torchvision.models as models
import os
from torch.nn import Module

"""
Template module for a base model submission to brain-score
"""


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def load_model(modelname='resnet', resume=None):
    if modelname == 'resnet':
        model = models.resnet50()
    else:
        raise ValueError("Architechture {} not valid.".format(modelname))
    # model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    checkpoint_file = resume
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    # model = model.module  # unwrap

    return model



# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)
dirname = os.path.dirname(__file__)
weights_path = os.path.join(dirname, 'saved-weights/model_best.pth.tar')
print(f"weights path is: {weights_path}")
model = load_model(resume=weights_path)
# import pdb
# pdb.set_trace()

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='r50-e35-cut1', model=model, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='r50-e35-cut1', activations_model=activations_model,
                        # specify layers to consider
                        layers=['layer1', 'layer2', 'layer3', 'layer4'])


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['r50-e35-cut1']


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
    # assert name == 'resnet-lr0.001'

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
    # assert name == 'resnet-lr0.001'

    # returns the layers you want to consider
    return ['layer1', 'layer2', 'layer3', 'layer4']

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
    # dirname = os.path.dirname(__file__)
    # print(dirname)
    # print (os.getcwd())
    check_models.check_base_models(__name__)
