from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['simclr_naive_tutorial'] = lambda: ModelCommitment(identifier='simclr_naive_tutorial', activations_model=get_model('simclr_naive_tutorial'), layers=get_layers('simclr_naive_tutorial'))
