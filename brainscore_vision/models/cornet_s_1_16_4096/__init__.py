from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_s_1_16_4096'] = lambda: ModelCommitment(identifier='cornet_s_1_16_4096', activations_model=get_model('cornet_s_1_16_4096'), layers=get_layers('cornet_s_1_16_4096'))
