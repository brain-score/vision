from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['half-width-r18-lc_wsh_100'] = ModelCommitment(identifier='half-width-r18-lc_wsh_100', activations_model=get_model('half-width-r18-lc_wsh_100'), layers=get_layers('half-width-r18-lc_wsh_100'))
