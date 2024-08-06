from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['efficientnet-b7'] = \
    lambda: ModelCommitment(identifier='efficientnet-b7', activations_model=get_model('efficientnet-b7'), layers=get_layers('efficientnet-b7'))