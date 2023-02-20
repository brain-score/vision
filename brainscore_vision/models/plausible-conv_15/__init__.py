from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['0.25xCORNet-S-LC'] = ModelCommitment(identifier='0.25xCORNet-S-LC', activations_model=get_model('0.25xCORNet-S-LC'), layers=get_layers('0.25xCORNet-S-LC'))
