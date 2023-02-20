from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetv2'] = ModelCommitment(identifier='effnetv2', activations_model=get_model('effnetv2'), layers=get_layers('effnetv2'))
