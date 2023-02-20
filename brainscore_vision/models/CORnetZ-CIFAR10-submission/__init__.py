from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnetZ_CIFAR10'] = ModelCommitment(identifier='CORnetZ_CIFAR10', activations_model=get_model('CORnetZ_CIFAR10'), layers=get_layers('CORnetZ_CIFAR10'))
