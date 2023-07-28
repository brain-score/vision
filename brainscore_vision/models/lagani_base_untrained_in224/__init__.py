from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-base-untrained-in224'] = ModelCommitment(identifier='lagani-base-untrained-in224', activations_model=get_model('lagani-base-untrained-in224'), layers=get_layers('lagani-base-untrained-in224'))
