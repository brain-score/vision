from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-architecture'] = ModelCommitment(identifier='lagani-architecture', activations_model=get_model('lagani-architecture'), layers=get_layers('lagani-architecture'))
