from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_aacn'] = ModelCommitment(identifier='resnet50_aacn', activations_model=get_model('resnet50_aacn'), layers=get_layers('resnet50_aacn'))
