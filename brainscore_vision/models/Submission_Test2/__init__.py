from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnetVanilla'] = ModelCommitment(identifier='CORnetVanilla', activations_model=get_model('CORnetVanilla'), layers=get_layers('CORnetVanilla'))
