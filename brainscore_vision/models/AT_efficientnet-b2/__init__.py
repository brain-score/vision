from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['AT_efficientnet-b2'] = \
    lambda: ModelCommitment(identifier='AT_efficientnet-b2', activations_model=get_model('AT_efficientnet-b2'), layers=get_layers('AT_efficientnet-b2'))
