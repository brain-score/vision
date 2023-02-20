from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['res2net'] = ModelCommitment(identifier='res2net', activations_model=get_model('res2net'), layers=get_layers('res2net'))
