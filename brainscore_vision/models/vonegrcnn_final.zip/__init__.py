from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vone+grcnn55'] = ModelCommitment(identifier='vone+grcnn55', activations_model=get_model('vone+grcnn55'), layers=get_layers('vone+grcnn55'))
