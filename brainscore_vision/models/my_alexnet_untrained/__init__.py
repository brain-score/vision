from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet-untrained'] = ModelCommitment(identifier='alexnet-untrained', activations_model=get_model('alexnet-untrained'), layers=get_layers('alexnet-untrained'))
