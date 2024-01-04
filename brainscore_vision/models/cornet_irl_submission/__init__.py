from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_irl'] = ModelCommitment(identifier='cornet_irl', activations_model=get_model('cornet_irl'), layers=get_layers('cornet_irl'))
