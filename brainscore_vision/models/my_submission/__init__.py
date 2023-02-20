from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['C-RN50'] = ModelCommitment(identifier='C-RN50', activations_model=get_model('C-RN50'), layers=get_layers('C-RN50'))
