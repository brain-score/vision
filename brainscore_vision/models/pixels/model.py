from collections import OrderedDict

import numpy as np
from PIL import Image

from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from brainscore_vision.model_helpers.check_submission import check_models

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
    return ['pixels']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'pixels'
    return PixelModel()


class PixelModel:
    def __init__(self):
        self._extractor = ActivationsExtractorHelper(identifier='pixels', preprocessing=None,
                                                     get_activations=self._pixels_from_paths)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def _pixels_from_paths(self, paths, layer_names):
        np.testing.assert_array_equal(layer_names, ['pixels'])
        pixels = [self._parse_image(path) for path in paths]
        return OrderedDict([('pixels', np.array(pixels))])

    def _parse_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')  # make sure everything is in RGB and not grayscale L
        image = image.resize((256, 256))  # resize all images to same size
        return np.array(image)


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
    assert name == 'pixels'
    return ['pixels']



def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the  BaeeModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
