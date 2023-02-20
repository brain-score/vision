from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornets-cifar10'] = ModelCommitment(identifier='cornets-cifar10', activations_model=get_model('cornets-cifar10'), layers=get_layers('cornets-cifar10'))
