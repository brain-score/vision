from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['hmax'] = ModelCommitment(identifier='hmax', activations_model=get_model('hmax'), layers=get_layers('hmax'))
