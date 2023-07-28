from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_aacn_8heads'] = ModelCommitment(identifier='resnet18_aacn_8heads', activations_model=get_model('resnet18_aacn_8heads'), layers=get_layers('resnet18_aacn_8heads'))
