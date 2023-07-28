from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['moco_101_70'] = ModelCommitment(identifier='moco_101_70', activations_model=get_model('moco_101_70'), layers=get_layers('moco_101_70'))
