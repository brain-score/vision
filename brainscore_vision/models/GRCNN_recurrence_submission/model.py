from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
import functools
from torch.nn import init
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from model_tools.check_submission import check_models

import os
import sys
mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(mydir)
from VOneRecurrenceCommitment import VOneRecurrenceCommitment
from GRCNN import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocessing = functools.partial(load_preprocess_images, image_size=224)

MODEL_NAME = 'my-model'
M = Model(num_classes=1000)
submitted_layers_names = [
    'VOneBlock.output','myGRCL1.A','myGRCL2.A','myGRCL3.A','myGRCL4.A'
]
activations_model = PytorchWrapper(
    identifier=MODEL_NAME, model=M, preprocessing=preprocessing)

# TODO: da scegliere
time_mapping = {'IT': {i: (time_bin_start, time_bin_start + 10)
                       for i, time_bin_start in enumerate(range(70, 250, 10))}}

model = VOneRecurrenceCommitment(
    identifier=MODEL_NAME, activations_model=activations_model,
    # specify layers to consider
    layers=submitted_layers_names,
    time_mapping=time_mapping)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return [MODEL_NAME]


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
    assert name == MODEL_NAME

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
    assert name == MODEL_NAME

    # returns the layers you want to consider
    return submitted_layers_names

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
