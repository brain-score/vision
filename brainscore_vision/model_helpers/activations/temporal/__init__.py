from ..core import ActivationsExtractorHelper
from .io.inputs import Video
from .io.outputs import Activation
from typing import List


class PytorchBaseModel:
    pass

class spec:
    pass


class TemporalActivationsExtractorHelper(ActivationsExtractorHelper):
    ### CAUTION: neet more rewrite here
    def __init__(self, base_model):
        self.base_model = base_model
        get_activations = self._build_get_activations()
        preprocessing = self._preprocessing
        super().__init__(identifier=base_model.__class__.__name__, get_activations=get_activations, preprocessing=preprocessing)

    def _build_get_activations(self):
        def get_activations(videos: List[Video], layer_names):
            activation = self.base_model(videos, layer_names)
            return activation.to_compact()
        return get_activations
    
    def _preprocessing(self, paths):
        return [Video(path) for path in paths]

    def __call__(self, stimuli, layers, stimuli_identifier=None):
        return super().__call__(stimuli, layers, stimuli_identifier)
    
    def _package_layer(self, layer_activations, layer, stimuli_paths):
        raise NotImplementedError()