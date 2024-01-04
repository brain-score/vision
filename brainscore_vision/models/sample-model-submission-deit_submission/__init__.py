from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['deit'] = ModelCommitment(identifier='deit', activations_model=get_model('deit'), layers=get_layers('deit'))
