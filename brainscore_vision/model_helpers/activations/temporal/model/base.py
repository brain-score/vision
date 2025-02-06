import numpy as np
from ..inputs.base import Stimulus
from ..core import ActivationsExtractor
from ..core import TemporalInferencer, Inferencer

from typing import List, Callable, Any, Dict


class ActivationWrapper:
    def __init__(
            self, 
            identifier : str, 
            preprocessing : Callable[[List[Stimulus]], Any],
            inferencer_cls : Inferencer = TemporalInferencer, 
            **extractor_kwargs
        ):
        self.identifier = identifier
        self.preprocessing = preprocessing
        self.build_extractor(inferencer_cls, **extractor_kwargs)

    # List[preprocessed_input] -> Dict[layer -> np.array]
    def get_activations(self, inputs : List[Stimulus], layers : List[str]) -> Dict[str, np.array]:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)
    
    def build_extractor(self, inferencer_cls, *args, **kwargs):
        extractor = ActivationsExtractor(identifier=self.identifier, 
                                         inferencer=inferencer_cls(get_activations=self.get_activations, 
                                                        preprocessing=self.preprocessing, *args, **kwargs))
        extractor.insert_attrs(self)
        self._extractor = extractor