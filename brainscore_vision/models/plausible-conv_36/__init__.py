from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-50_untrained'] = ModelCommitment(identifier='resnet-50_untrained', activations_model=get_model('resnet-50_untrained'), layers=get_layers('resnet-50_untrained'))
