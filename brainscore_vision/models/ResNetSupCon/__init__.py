from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ResNetConSup'] = ModelCommitment(identifier='ResNetConSup', activations_model=get_model('ResNetConSup'), layers=get_layers('ResNetConSup'))
