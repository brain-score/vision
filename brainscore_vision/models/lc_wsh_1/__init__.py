from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['0.5x_resnet-18_LC_w_sh_1_iter'] = ModelCommitment(identifier='0.5x_resnet-18_LC_w_sh_1_iter', activations_model=get_model('0.5x_resnet-18_LC_w_sh_1_iter'), layers=get_layers('0.5x_resnet-18_LC_w_sh_1_iter'))
