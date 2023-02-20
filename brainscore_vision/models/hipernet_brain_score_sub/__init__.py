from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['hipernet'] = ModelCommitment(identifier='hipernet', activations_model=get_model('hipernet'), layers=get_layers('hipernet'))
