from brainscore_vision import model_registry
from .helpers.helpers import CORnetCommitment, _build_time_mappings
from .model import get_model, get_layers, TIME_MAPPINGS


model_registry['cornet_s'] = lambda: CORnetCommitment(identifier='cornet_s', activations_model=get_model('cornet_s'),
                                                      layers=get_layers('cornet_s'),
                                                      time_mapping=_build_time_mappings(TIME_MAPPINGS))