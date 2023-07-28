from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dorinet'] = ModelCommitment(identifier='dorinet', activations_model=get_model('dorinet'), layers=get_layers('dorinet'))
