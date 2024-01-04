from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_lr0.3_epoch60'] = ModelCommitment(identifier='resnet18_lr0.3_epoch60', activations_model=get_model('resnet18_lr0.3_epoch60'), layers=get_layers('resnet18_lr0.3_epoch60'))
