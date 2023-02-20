from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['hmax_standard'] = ModelCommitment(identifier='hmax_standard', activations_model=get_model('hmax_standard'), layers=get_layers('hmax_standard'))
