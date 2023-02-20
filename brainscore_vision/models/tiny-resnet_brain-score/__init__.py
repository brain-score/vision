from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_tiny-mnist'] = ModelCommitment(identifier='resnet_tiny-mnist', activations_model=get_model('resnet_tiny-mnist'), layers=get_layers('resnet_tiny-mnist'))
