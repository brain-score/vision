from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetv2m'] = ModelCommitment(identifier='effnetv2m', activations_model=get_model('effnetv2m'), layers=get_layers('effnetv2m'))
