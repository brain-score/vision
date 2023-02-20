from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn'] = ModelCommitment(identifier='grcnn', activations_model=get_model('grcnn'), layers=get_layers('grcnn'))
