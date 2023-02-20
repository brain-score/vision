from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['moco_full'] = ModelCommitment(identifier='moco_full', activations_model=get_model('moco_full'), layers=get_layers('moco_full'))
