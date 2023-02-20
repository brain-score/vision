import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""

MODEL_NAME = 'alexnet-untrained'

def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return [MODEL_NAME]


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
    model = torchvision.models.alexnet(pretrained=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=MODEL_NAME, model=model, preprocessing=preprocessing)
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
    assert name == MODEL_NAME
    return ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
            'classifier.2', 'classifier.5']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@incollection{NIPS2012_4824,
                title = {ImageNet Classification with Deep Convolutional Neural Networks},
                author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                booktitle = {Advances in Neural Information Processing Systems 25},
                editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
                pages = {1097--1105},
                year = {2012},
                publisher = {Curran Associates, Inc.},
                url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
                }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

