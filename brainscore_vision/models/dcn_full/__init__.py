from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dcn_full_mar15'] = ModelCommitment(identifier='dcn_full_mar15', activations_model=get_model('dcn_full_mar15'), layers=get_layers('dcn_full_mar15'))
