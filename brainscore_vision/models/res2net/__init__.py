from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['res2net'] = lambda: ModelCommitment(identifier='res2net', activations_model=get_model('res2net'), layers=get_layers('res2net'))
