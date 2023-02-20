from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_test46'] = ModelCommitment(identifier='alexnet_test46', activations_model=get_model('alexnet_test46'), layers=get_layers('alexnet_test46'))
