from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-lr0.1'] = ModelCommitment(identifier='resnet-lr0.1', activations_model=get_model('resnet-lr0.1'), layers=get_layers('resnet-lr0.1'))
