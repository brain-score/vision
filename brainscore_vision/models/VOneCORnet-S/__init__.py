from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model
from .helpers.vonecornets import VOneCORnetCommitment, _build_time_mappings, CORNET_S_TIMEMAPPING


model_registry['VOneCORnet-S'] = \
    lambda: VOneCORnetCommitment(identifier='VOneCORnet-S', activations_model=get_model('VOneCORnet-S'),
                                layers=get_layers('VOneCORnet-S'), time_mapping=_build_time_mappings(CORNET_S_TIMEMAPPING))