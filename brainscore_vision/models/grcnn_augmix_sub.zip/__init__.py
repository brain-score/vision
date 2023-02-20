from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_robust'] = ModelCommitment(identifier='grcnn_robust', activations_model=get_model('grcnn_robust'), layers=get_layers('grcnn_robust'))
