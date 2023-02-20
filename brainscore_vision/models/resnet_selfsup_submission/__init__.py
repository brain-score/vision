from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-simclr'] = ModelCommitment(identifier='resnet50-simclr', activations_model=get_model('resnet50-simclr'), layers=get_layers('resnet50-simclr'))
