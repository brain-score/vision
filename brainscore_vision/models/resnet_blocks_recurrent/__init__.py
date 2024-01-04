from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_recur_adv_exp4_jun10'] = ModelCommitment(identifier='resnet_recur_adv_exp4_jun10', activations_model=get_model('resnet_recur_adv_exp4_jun10'), layers=get_layers('resnet_recur_adv_exp4_jun10'))
