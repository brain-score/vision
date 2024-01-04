from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-sup'] = ModelCommitment(identifier='resnet50-sup', activations_model=get_model('resnet50-sup'), layers=get_layers('resnet50-sup'))
