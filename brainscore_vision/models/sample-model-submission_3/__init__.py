from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['RN50x4'] = ModelCommitment(identifier='RN50x4', activations_model=get_model('RN50x4'), layers=get_layers('RN50x4'))
