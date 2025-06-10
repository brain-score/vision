from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['kap2'] = lambda: ModelCommitment(identifier='kap2', activations_model=get_model('kap2'), layers=get_layers('kap2'))
