from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['skgrcnn'] = ModelCommitment(identifier='skgrcnn', activations_model=get_model('skgrcnn'), layers=get_layers('skgrcnn'))
