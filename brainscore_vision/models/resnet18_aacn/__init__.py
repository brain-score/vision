from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_aacn6'] = ModelCommitment(identifier='resnet18_aacn6', activations_model=get_model('resnet18_aacn6'), layers=get_layers('resnet18_aacn6'))
