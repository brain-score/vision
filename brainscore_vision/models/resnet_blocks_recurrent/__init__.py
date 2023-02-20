from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet'] = ModelCommitment(identifier='cornet', activations_model=get_model('cornet'), layers=get_layers('cornet'))
