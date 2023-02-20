from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['feedback-alignment-model2'] = ModelCommitment(identifier='feedback-alignment-model2', activations_model=get_model('feedback-alignment-model2'), layers=get_layers('feedback-alignment-model2'))
