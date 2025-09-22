from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['sewresnet101'] = lambda: ModelCommitment(identifier='sewresnet101', activations_model=get_model('sewresnet101'), layers=get_layers('sewresnet101'))