from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornets-dvs'] = ModelCommitment(identifier='cornets-dvs', activations_model=get_model('cornets-dvs'), layers=get_layers('cornets-dvs'))
