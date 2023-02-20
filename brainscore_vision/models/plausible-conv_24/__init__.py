from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18-LC_conv_init_m'] = ModelCommitment(identifier='resnet-18-LC_conv_init_m', activations_model=get_model('resnet-18-LC_conv_init_m'), layers=get_layers('resnet-18-LC_conv_init_m'))
model_registry['resnet-18-LC_w_sh_1_iter_conv_init_m'] = ModelCommitment(identifier='resnet-18-LC_w_sh_1_iter_conv_init_m', activations_model=get_model('resnet-18-LC_w_sh_1_iter_conv_init_m'), layers=get_layers('resnet-18-LC_w_sh_1_iter_conv_init_m'))
model_registry['resnet-18-LC_w_sh_10_iter_conv_init_m'] = ModelCommitment(identifier='resnet-18-LC_w_sh_10_iter_conv_init_m', activations_model=get_model('resnet-18-LC_w_sh_10_iter_conv_init_m'), layers=get_layers('resnet-18-LC_w_sh_10_iter_conv_init_m'))
model_registry['resnet-18-LC_w_sh_100_iter_conv_init_m'] = ModelCommitment(identifier='resnet-18-LC_w_sh_100_iter_conv_init_m', activations_model=get_model('resnet-18-LC_w_sh_100_iter_conv_init_m'), layers=get_layers('resnet-18-LC_w_sh_100_iter_conv_init_m'))
model_registry['resnet-18-LC_d_w_sh_1x1_conv_init_m'] = ModelCommitment(identifier='resnet-18-LC_d_w_sh_1x1_conv_init_m', activations_model=get_model('resnet-18-LC_d_w_sh_1x1_conv_init_m'), layers=get_layers('resnet-18-LC_d_w_sh_1x1_conv_init_m'))
