from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_152_v1'] = lambda: ModelCommitment(identifier='resnet_152_v1', activations_model=get_model('resnet_152_v1'), layers=get_layers('resnet_152_v1'))