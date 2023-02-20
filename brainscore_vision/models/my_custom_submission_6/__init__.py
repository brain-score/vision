from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['test1'] = ModelCommitment(identifier='test1', activations_model=get_model('test1'), layers=get_layers('test1'))
