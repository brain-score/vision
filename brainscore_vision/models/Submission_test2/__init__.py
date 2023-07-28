from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnetCustom'] = ModelCommitment(identifier='CORnetCustom', activations_model=get_model('CORnetCustom'), layers=get_layers('CORnetCustom'))
