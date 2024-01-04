from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vone_alexnet'] = ModelCommitment(identifier='vone_alexnet', activations_model=get_model('vone_alexnet'), layers=get_layers('vone_alexnet'))
