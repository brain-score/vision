from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['RN101'] = ModelCommitment(identifier='RN101', activations_model=get_model('RN101'), layers=get_layers('RN101'))
