from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-10W-two-blocks-LC_prelim'] = ModelCommitment(identifier='resnet-10W-two-blocks-LC_prelim', activations_model=get_model('resnet-10W-two-blocks-LC_prelim'), layers=get_layers('resnet-10W-two-blocks-LC_prelim'))
model_registry['resnet-10Wm-two-blocks-LC_prelim'] = ModelCommitment(identifier='resnet-10Wm-two-blocks-LC_prelim', activations_model=get_model('resnet-10Wm-two-blocks-LC_prelim'), layers=get_layers('resnet-10Wm-two-blocks-LC_prelim'))
model_registry['resnet-18-LC_d_w_sh_conv_init_prelim'] = ModelCommitment(identifier='resnet-18-LC_d_w_sh_conv_init_prelim', activations_model=get_model('resnet-18-LC_d_w_sh_conv_init_prelim'), layers=get_layers('resnet-18-LC_d_w_sh_conv_init_prelim'))
