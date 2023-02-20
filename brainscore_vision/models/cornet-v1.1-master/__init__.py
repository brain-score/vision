from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet-v1.1'] = ModelCommitment(identifier='cornet-v1.1', activations_model=get_model('cornet-v1.1'), layers=get_layers('cornet-v1.1'))
