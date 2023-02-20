from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ode_net_apr18'] = ModelCommitment(identifier='ode_net_apr18', activations_model=get_model('ode_net_apr18'), layers=get_layers('ode_net_apr18'))
