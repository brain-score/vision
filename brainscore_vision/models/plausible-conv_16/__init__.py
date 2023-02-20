from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['0.25xCORNet-S-LC_conv_init'] = ModelCommitment(identifier='0.25xCORNet-S-LC_conv_init', activations_model=get_model('0.25xCORNet-S-LC_conv_init'), layers=get_layers('0.25xCORNet-S-LC_conv_init'))
