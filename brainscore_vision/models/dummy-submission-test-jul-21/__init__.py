from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet'] = ModelCommitment(identifier='alexnet', activations_model=get_model('alexnet'), layers=get_layers('alexnet'))
