import xarray as xr
from ..inputs.base import Stimulus
from ..core import TemporalExtractorHelper

from typing import List


class ActivationWrapper:
    def __init__(self, identifier, spec, preprocessing, *args, exectractor_cls=TemporalExtractorHelper, **kwargs):
        self.identifier = identifier
        self.spec = spec
        self._extractor = self._build_extractor(exectractor_cls, self.get_activations, preprocessing, spec, *args, **kwargs)

    # List[preprocessed_input] -> xr.DataArray
    def get_activations(self, inputs, layers: List[str]):
        return xr.DataArray()

    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)
    
    def _build_extractor(self, exectractor_cls, get_activations, preprocessing, spec, *args, **kwargs):
        extractor = exectractor_cls(identifier=self.identifier, get_activations=get_activations, 
                                    preprocessing=preprocessing, spec=spec, *args, **kwargs)
        extractor.insert_attrs(self)
        return extractor
    
    @property
    def is_specified(self):
        return self.spec is not None
    
    @property
    def specified_layers(self):
        assert self.is_specified, "spec is not set"
        return list(self.spec['activation'].keys())