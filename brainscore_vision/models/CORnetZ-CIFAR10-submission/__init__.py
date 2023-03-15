from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnetZ_CIFAR10_bs32_20_04'] = ModelCommitment(identifier='CORnetZ_CIFAR10_bs32_20_04', activations_model=get_model('CORnetZ_CIFAR10_bs32_20_04'), layers=get_layers('CORnetZ_CIFAR10_bs32_20_04'))
