from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['densenet-169'] = \
    lambda: ModelCommitment(identifier='densenet-169', activations_model=get_model('densenet-169'), layers=get_layers('densenet-169'))