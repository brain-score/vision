from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vonegrcnn_52e_full'] = ModelCommitment(identifier='vonegrcnn_52e_full', activations_model=get_model('vonegrcnn_52e_full'), layers=get_layers('vonegrcnn_52e_full'))
