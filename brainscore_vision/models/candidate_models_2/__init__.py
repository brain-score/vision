from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vgg-16'] = ModelCommitment(identifier='vgg-16', activations_model=get_model('vgg-16'), layers=get_layers('vgg-16'))
model_registry['resnet-101_v1'] = ModelCommitment(identifier='resnet-101_v1', activations_model=get_model('resnet-101_v1'), layers=get_layers('resnet-101_v1'))
