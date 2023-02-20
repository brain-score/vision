from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CornetVanilla2'] = ModelCommitment(identifier='CornetVanilla2', activations_model=get_model('CornetVanilla2'), layers=get_layers('CornetVanilla2'))
