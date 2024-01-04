from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vone_grcnn_l2'] = ModelCommitment(identifier='vone_grcnn_l2', activations_model=get_model('vone_grcnn_l2'), layers=get_layers('vone_grcnn_l2'))
