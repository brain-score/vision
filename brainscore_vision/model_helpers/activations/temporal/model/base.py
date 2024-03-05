import xarray as xr
from ..inputs.base import Stimulus
from ..core import ActivationsExtractor
from ..core.video import TemporalInferencer

from typing import List


class ActivationWrapper:
    def __init__(self, identifier, preprocessing, inferencer_cls=TemporalInferencer, **extractor_kwargs):
        self.identifier = identifier
        self.preprocessing = preprocessing
        self.build_extractor(inferencer_cls, **extractor_kwargs)

    # List[preprocessed_input] -> xr.DataArray
    def get_activations(self, inputs, layers: List[str]):
        return xr.DataArray()

    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)
    
    def build_extractor(self, inferencer_cls, *args, **kwargs):
        extractor = ActivationsExtractor(identifier=self.identifier, 
                                         inferencer=inferencer_cls(get_activations=self.get_activations, 
                                                        preprocessing=self.preprocessing, *args, **kwargs))
        extractor.insert_attrs(self)
        self._extractor = extractor