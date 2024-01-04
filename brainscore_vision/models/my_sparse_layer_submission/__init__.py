from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['my-model'] = ModelCommitment(identifier='my-model', activations_model=get_model('my-model'), layers=get_layers('my-model'))
