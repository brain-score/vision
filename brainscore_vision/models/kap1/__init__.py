from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['kap1'] = lambda: ModelCommitment(identifier='kap1', activations_model=get_model('kap1'), layers=get_layers('kap1'))
