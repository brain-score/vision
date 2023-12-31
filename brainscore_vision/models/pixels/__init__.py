from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['pixels'] = lambda: ModelCommitment(identifier='pixels', activations_model=get_model('pixels'), layers=get_layers('pixels'))