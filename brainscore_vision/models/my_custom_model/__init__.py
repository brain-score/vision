from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['my_custom_model'] = lambda: ModelCommitment(identifier='my_custom_model', activations_model=get_model('my_custom_model'), layers=get_layers('my_custom_model'))
