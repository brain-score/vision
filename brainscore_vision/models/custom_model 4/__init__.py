from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['skgcrnn'] = ModelCommitment(identifier='skgcrnn', activations_model=get_model('skgcrnn'), layers=get_layers('skgcrnn'))
