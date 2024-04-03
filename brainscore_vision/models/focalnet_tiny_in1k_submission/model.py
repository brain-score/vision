from model_tools.check_submission import check_models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import torchvision
import functools
import torch
import os
import timm

dir_path = os.path.dirname(os.path.realpath(__file__))

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
    return ['focalnet_tiny_lrf_in1k']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'focalnet_tiny_lrf_in1k'

    model = timm.create_model('focalnet_tiny_lrf.ms_in1k', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='focalnet_tiny_lrf_in1k', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'focalnet_tiny_lrf_in1k'
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return ['layers.0.blocks.0', 'layers.0.blocks.1', 'layers.1.downsample', 
        'layers.1.blocks.0', 'layers.1.blocks.1', 'layers.2.downsample', 'layers.2.blocks.0',
        'layers.2.blocks.1', 'layers.2.blocks.2', 'layers.2.blocks.3', 'layers.2.blocks.4',
         'layers.2.blocks.5', 'layers.3.downsample', 'layers.3.blocks.0', 'layers.3.blocks.1', 'norm','head.global_pool', 'head.fc']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)