from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18_test_m'] = ModelCommitment(identifier='resnet-18_test_m', activations_model=get_model('resnet-18_test_m'), layers=get_layers('resnet-18_test_m'))
