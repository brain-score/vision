from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18_LC_w_sh_1_iter_test'] = ModelCommitment(identifier='resnet-18_LC_w_sh_1_iter_test', activations_model=get_model('resnet-18_LC_w_sh_1_iter_test'), layers=get_layers('resnet-18_LC_w_sh_1_iter_test'))
