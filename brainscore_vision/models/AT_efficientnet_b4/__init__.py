from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['AT_efficientnet-b4'] = \
    lambda: ModelCommitment(identifier='AT_efficientnet-b4', activations_model=get_model('AT_efficientnet-b4'), layers=get_layers('AT_efficientnet-b4'))
