from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet101_full'] = ModelCommitment(identifier='resnet101_full', activations_model=get_model('resnet101_full'), layers=get_layers('resnet101_full'))
