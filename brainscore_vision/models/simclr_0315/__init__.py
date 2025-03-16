from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['simclr_0315'] = lambda: ModelCommitment(identifier='simclr_0315', activations_model=get_model('simclr_0315'), layers=get_layers('simclr_0315'))
