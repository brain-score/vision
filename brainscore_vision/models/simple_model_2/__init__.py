from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['simple-baseline-model2'] = ModelCommitment(identifier='simple-baseline-model2', activations_model=get_model('simple-baseline-model2'), layers=get_layers('simple-baseline-model2'))
