from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18-LC_untrained'] = ModelCommitment(identifier='resnet-18-LC_untrained', activations_model=get_model('resnet-18-LC_untrained'), layers=get_layers('resnet-18-LC_untrained'))
model_registry['resnet-18_untrained'] = ModelCommitment(identifier='resnet-18_untrained', activations_model=get_model('resnet-18_untrained'), layers=get_layers('resnet-18_untrained'))
