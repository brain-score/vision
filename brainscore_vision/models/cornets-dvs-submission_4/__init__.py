from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vonecornets-cifar10'] = ModelCommitment(identifier='vonecornets-cifar10', activations_model=get_model('vonecornets-cifar10'), layers=get_layers('vonecornets-cifar10'))
