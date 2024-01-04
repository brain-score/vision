from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_test'] = ModelCommitment(identifier='resnet50_test', activations_model=get_model('resnet50_test'), layers=get_layers('resnet50_test'))
