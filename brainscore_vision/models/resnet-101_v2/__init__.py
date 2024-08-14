from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-101_v2'] = lambda: ModelCommitment(identifier='resnet-101_v2', activations_model=get_model('resnet-101_v2'), layers=get_layers('resnet-101_v2'))
