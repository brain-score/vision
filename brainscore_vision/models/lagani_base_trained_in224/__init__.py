from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-base-trained-in224'] = ModelCommitment(identifier='lagani-base-trained-in224', activations_model=get_model('lagani-base-trained-in224'), layers=get_layers('lagani-base-trained-in224'))
