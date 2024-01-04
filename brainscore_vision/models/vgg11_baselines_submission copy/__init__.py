from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['neg_samples_e2e'] = ModelCommitment(identifier='neg_samples_e2e', activations_model=get_model('neg_samples_e2e'), layers=get_layers('neg_samples_e2e'))
