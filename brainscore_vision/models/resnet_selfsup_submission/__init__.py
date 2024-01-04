from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-barlow'] = ModelCommitment(identifier='resnet50-barlow', activations_model=get_model('resnet50-barlow'), layers=get_layers('resnet50-barlow'))
