from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornets-cifar10dvs'] = ModelCommitment(identifier='cornets-cifar10dvs', activations_model=get_model('cornets-cifar10dvs'), layers=get_layers('cornets-cifar10dvs'))
