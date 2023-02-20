from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_untrained'] = ModelCommitment(identifier='alexnet_untrained', activations_model=get_model('alexnet_untrained'), layers=get_layers('alexnet_untrained'))
