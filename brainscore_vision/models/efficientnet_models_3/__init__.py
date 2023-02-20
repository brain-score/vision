from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['AT_efficientnet-b0'] = ModelCommitment(identifier='AT_efficientnet-b0', activations_model=get_model('AT_efficientnet-b0'), layers=get_layers('AT_efficientnet-b0'))
