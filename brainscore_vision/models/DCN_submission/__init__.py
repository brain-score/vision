from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dcn_ode'] = ModelCommitment(identifier='dcn_ode', activations_model=get_model('dcn_ode'), layers=get_layers('dcn_ode'))
