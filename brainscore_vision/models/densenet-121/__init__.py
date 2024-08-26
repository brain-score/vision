from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['densenet-121'] = \
    lambda: ModelCommitment(identifier='densenet-121', activations_model=get_model('densenet-121'), layers=get_layers('densenet-121'))