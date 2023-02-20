from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet101'] = ModelCommitment(identifier='resnet101', activations_model=get_model('resnet101'), layers=get_layers('resnet101'))
