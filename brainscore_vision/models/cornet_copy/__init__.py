from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_copy'] = lambda: ModelCommitment(identifier='cornet_copy', activations_model=get_model('cornet_copy'), layers=get_layers('cornet_copy'))
