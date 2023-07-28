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
import torchvision.models as models


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['my-model']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.

def get_model(name):
    assert name == 'my-model'
    model = models.resnet18()
    model.fc = nn.Linear(512, 10) #for cifar10 training
    model.load_state_dict(torch.load('best_resnet18.pth'))
    model.fc = nn.Linear(512, 1000)  # imagenet output mapping
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing)
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
    assert name == 'my-model'

    # returns the layers you want to consider
    return  ['conv1', 'layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']

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

