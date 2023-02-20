from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['moco'] = ModelCommitment(identifier='moco', activations_model=get_model('moco'), layers=get_layers('moco'))
