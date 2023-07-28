from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-cifar'] = ModelCommitment(identifier='resnet50-cifar', activations_model=get_model('resnet50-cifar'), layers=get_layers('resnet50-cifar'))
