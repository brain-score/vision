from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['expansion'] = lambda: ModelCommitment(identifier='expansion', activations_model=get_model('expansion'), layers=get_layers('expansion'))