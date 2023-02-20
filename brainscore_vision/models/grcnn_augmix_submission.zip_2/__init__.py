from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_robust_v2'] = ModelCommitment(identifier='grcnn_robust_v2', activations_model=get_model('grcnn_robust_v2'), layers=get_layers('grcnn_robust_v2'))
