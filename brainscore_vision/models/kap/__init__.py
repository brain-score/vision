from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['kap'] = lambda: ModelCommitment(identifier='kap', activations_model=get_model('kap'), layers=get_layers('kap'))
