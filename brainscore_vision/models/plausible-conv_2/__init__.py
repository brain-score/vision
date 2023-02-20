from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['0.25xCORNet-S_LC_1st_conv_1x1_conv_preliminary'] = ModelCommitment(identifier='0.25xCORNet-S_LC_1st_conv_1x1_conv_preliminary', activations_model=get_model('0.25xCORNet-S_LC_1st_conv_1x1_conv_preliminary'), layers=get_layers('0.25xCORNet-S_LC_1st_conv_1x1_conv_preliminary'))
