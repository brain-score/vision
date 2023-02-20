from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['custom_model_v1'] = ModelCommitment(identifier='custom_model_v1', activations_model=get_model('custom_model_v1'), layers=get_layers('custom_model_v1'))
