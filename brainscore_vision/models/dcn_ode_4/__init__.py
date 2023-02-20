from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dcn_ode_apr19_tinyImagenet'] = ModelCommitment(identifier='dcn_ode_apr19_tinyImagenet', activations_model=get_model('dcn_ode_apr19_tinyImagenet'), layers=get_layers('dcn_ode_apr19_tinyImagenet'))
