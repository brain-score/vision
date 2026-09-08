from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry['vonealexnet'] = lambda: ModelCommitment(
    identifier='vonealexnet',
    activations_model=get_model('vonealexnet'),
    layers=get_layers('vonealexnet'))
