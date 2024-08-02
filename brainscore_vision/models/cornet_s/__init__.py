from brainscore_vision import model_registry
from .helpers.helpers import CORnetCommitment, _build_time_mappings
from .model import get_model, get_layers, TIME_MAPPINGS


model_registry['CORnet-S'] = lambda: CORnetCommitment(identifier='CORnet-S', activations_model=get_model('CORnet-S'),
                                                      layers=get_layers('CORnet-S'),
                                                      time_mapping=_build_time_mappings(TIME_MAPPINGS))