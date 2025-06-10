from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['kapv1'] = lambda: ModelCommitment(identifier='kapv1', activations_model=get_model('kapv1'), layers=get_layers('kapv1'))