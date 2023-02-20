from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['my-custom-model'] = ModelCommitment(identifier='my-custom-model', activations_model=get_model('my-custom-model'), layers=get_layers('my-custom-model'))
