from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ode_net_mar31_tinyImagenet'] = ModelCommitment(identifier='ode_net_mar31_tinyImagenet', activations_model=get_model('ode_net_mar31_tinyImagenet'), layers=get_layers('ode_net_mar31_tinyImagenet'))
