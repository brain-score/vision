from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['AT_efficientnet-b7'] = ModelCommitment(identifier='AT_efficientnet-b7', activations_model=get_model('AT_efficientnet-b7'), layers=get_layers('AT_efficientnet-b7'))
model_registry['AT_efficientnet-b8'] = ModelCommitment(identifier='AT_efficientnet-b8', activations_model=get_model('AT_efficientnet-b8'), layers=get_layers('AT_efficientnet-b8'))
