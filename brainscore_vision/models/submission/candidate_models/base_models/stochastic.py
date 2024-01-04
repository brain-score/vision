
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.core import ActivationsExtractorHelper
from candidate_models.base_models.cornet import TemporalPytorchWrapper, TemporalExtractor

class StochasticPytorchWrapper(PytorchWrapper):
    def _build_extractor(self, identifier, preprocessing, get_activations, *args, **kwargs):
        return StochasticActivationsExtractorHelper(
            identifier=identifier, get_activations=get_activations, preprocessing=preprocessing,
            *args, **kwargs)


class StochasticActivationsExtractorHelper(ActivationsExtractorHelper):
    def _reduce_paths(self, stimuli_paths):
        return stimuli_paths

    def _expand_paths(self, activations, original_paths):
        return activations


class StochasticTemporalPytorchWrapper(TemporalPytorchWrapper):
    def _build_extractor(self, *args, **kwargs):
        if self._separate_time:
            return StochasticTemporalExtractor(*args, **kwargs)
        else:
            return super(StochasticTemporalPytorchWrapper, self)._build_extractor(*args, **kwargs)


class StochasticTemporalExtractor(TemporalExtractor):
    def _reduce_paths(self, stimuli_paths):
        return stimuli_paths

    def _expand_paths(self, activations, original_paths):
        return activations
