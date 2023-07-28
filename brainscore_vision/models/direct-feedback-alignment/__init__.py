from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cb-direct-feedback-alignment'] = ModelCommitment(identifier='cb-direct-feedback-alignment', activations_model=get_model('cb-direct-feedback-alignment'), layers=get_layers('cb-direct-feedback-alignment'))
