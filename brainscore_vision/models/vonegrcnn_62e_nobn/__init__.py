from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vonegrcnn_62e'] = ModelCommitment(identifier='vonegrcnn_62e', activations_model=get_model('vonegrcnn_62e'), layers=get_layers('vonegrcnn_62e'))
