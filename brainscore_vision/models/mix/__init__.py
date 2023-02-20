from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet'] = ModelCommitment(identifier='alexnet', activations_model=get_model('alexnet'), layers=get_layers('alexnet'))
model_registry['googlenet'] = ModelCommitment(identifier='googlenet', activations_model=get_model('googlenet'), layers=get_layers('googlenet'))
model_registry['resnet18'] = ModelCommitment(identifier='resnet18', activations_model=get_model('resnet18'), layers=get_layers('resnet18'))
