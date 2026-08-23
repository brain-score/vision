from collections import OrderedDict

import numpy as np

from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from .descriptor import GaborEntropyConfig, MeanAbsFoveatedGaborExtractor


MODEL_IDENTIFIER = 'gabor_filter_mean_abs_4x4_v2'
LAYER_NAME = 'gabor_mean_abs_foveated32_48_64'


class GaborFilterMeanAbsModel:
    def __init__(self, identifier=MODEL_IDENTIFIER, config=None):
        self.config = config or GaborEntropyConfig()
        self._descriptor = MeanAbsFoveatedGaborExtractor(self.config)
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier,
            preprocessing=None,
            get_activations=self._get_activations)
        self._extractor.insert_attrs(self)
        self.image_size = self.config.image_size

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)

    def _get_activations(self, paths, layer_names):
        np.testing.assert_array_equal(layer_names, [LAYER_NAME])
        features = self._descriptor.extract_from_paths(paths)
        return OrderedDict([(LAYER_NAME, features)])


def get_model(name):
    assert name == MODEL_IDENTIFIER
    return GaborFilterMeanAbsModel(identifier=MODEL_IDENTIFIER)


def get_layers(name):
    assert name == MODEL_IDENTIFIER
    return [LAYER_NAME]


def get_bibtex(model_identifier):
    return """"""
