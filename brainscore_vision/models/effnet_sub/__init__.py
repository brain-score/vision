from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb0'] = ModelCommitment(identifier='effnetb0', activations_model=get_model('effnetb0'), layers=get_layers('effnetb0'))
