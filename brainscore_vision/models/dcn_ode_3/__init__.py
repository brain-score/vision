from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dcn_ode_apr18'] = ModelCommitment(identifier='dcn_ode_apr18', activations_model=get_model('dcn_ode_apr18'), layers=get_layers('dcn_ode_apr18'))
