from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_MM'] = ModelCommitment(identifier='resnet_MM', activations_model=get_model('resnet_MM'), layers=get_layers('resnet_MM'))
