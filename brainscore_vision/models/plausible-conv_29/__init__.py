from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18-LC_d_w_sh_conv_init_prelim3'] = ModelCommitment(identifier='resnet-18-LC_d_w_sh_conv_init_prelim3', activations_model=get_model('resnet-18-LC_d_w_sh_conv_init_prelim3'), layers=get_layers('resnet-18-LC_d_w_sh_conv_init_prelim3'))
