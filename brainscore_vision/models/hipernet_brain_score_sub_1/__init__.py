from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['hipernet1'] = ModelCommitment(identifier='hipernet1', activations_model=get_model('hipernet1'), layers=get_layers('hipernet1'))
