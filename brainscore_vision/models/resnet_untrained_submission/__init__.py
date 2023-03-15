from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-untrained3'] = ModelCommitment(identifier='resnet50-untrained3', activations_model=get_model('resnet50-untrained3'), layers=get_layers('resnet50-untrained3'))
