from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornetz_contrastive'] = ModelCommitment(identifier='cornetz_contrastive', activations_model=get_model('cornetz_contrastive'), layers=get_layers('cornetz_contrastive'))
