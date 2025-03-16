from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['simclr_ver2'] = lambda: ModelCommitment(identifier='simclr_ver2', activations_model=get_model('simclr_ver2'), layers=get_layers('simclr_ver2'))
