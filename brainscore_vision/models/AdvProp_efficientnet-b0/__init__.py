from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['AdvProp_efficientnet-b0'] = \
    lambda: ModelCommitment(identifier='AdvProp_efficientnet-b0', activations_model=get_model('AdvProp_efficientnet-b0'), layers=get_layers('AdvProp_efficientnet-b0'))