from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cornetz_contrastive'] = lambda: ModelCommitment(identifier='cornetz_contrastive', activations_model=get_model('cornetz_contrastive'), layers=get_layers('cornetz_contrastive'))
