from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vone_grcnn_ll'] = ModelCommitment(identifier='vone_grcnn_ll', activations_model=get_model('vone_grcnn_ll'), layers=get_layers('vone_grcnn_ll'))
