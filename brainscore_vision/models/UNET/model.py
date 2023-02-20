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

"""
Template module for a base model submission to brain-score
"""


# define the UNET model:
unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')

preprocessing = functools.partial(load_preprocess_images, image_size=224)

# wrap the model to get layer activations
activations_model = PytorchWrapper(identifier='unet', model=unet, preprocessing=preprocessing)

# init the model
unet = ModelCommitment(identifier='unet', activations_model=activations_model,
                        layers=['inc.double_conv.2', 'down1.maxpool_conv.1.double_conv.2', 'up1.conv.double_conv.2'])


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    # from pytorch.py:
    return ['unet']



def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'unet'

    # link the custom model to the wrapper object:
    wrapper = activations_model
    wrapper.image_size = 224
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

    # from pytorch.py:
    assert name == 'unet'
    return ['inc.double_conv.2', 'down1.maxpool_conv.1.double_conv.2', 'up1.conv.double_conv.2']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{ronneberger2015u,
            title={U-net: Convolutional networks for biomedical image segmentation},
              author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
              booktitle={International Conference on Medical image computing and computer-assisted intervention},
              pages={234--241},
              year={2015},
              organization={Springer}
            }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
