from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['nasnet_mobile'] = \
    lambda: ModelCommitment(identifier='nasnet_mobile', activations_model=get_model('nasnet_mobile'), layers=get_layers('nasnet_mobile'))