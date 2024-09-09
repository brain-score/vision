from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet-18_test_m'] = \
    lambda: ModelCommitment(identifier='resnet-18_test_m', activations_model=get_model('resnet-18_test_m'), layers=get_layers('resnet-18_test_m'))


