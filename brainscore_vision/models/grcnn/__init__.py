from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['grcnn'] = \
    lambda: ModelCommitment(identifier='grcnn', activations_model=get_model('grcnn'), layers=get_layers('grcnn'))