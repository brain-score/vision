from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vonecornets-cifar10dvs'] = ModelCommitment(identifier='vonecornets-cifar10dvs', activations_model=get_model('vonecornets-cifar10dvs'), layers=get_layers('vonecornets-cifar10dvs'))
