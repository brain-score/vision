from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_s_2_14_0'] = lambda: ModelCommitment(identifier='cornet_s_2_14_0', activations_model=get_model('cornet_s_2_14_0'), layers=get_layers('cornet_s_2_14_0'))
