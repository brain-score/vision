from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_s'] = ModelCommitment(identifier='cornet_s', activations_model=get_model('cornet_s'), layers=get_layers('cornet_s'))
model_registry['vgg_19'] = ModelCommitment(identifier='vgg_19', activations_model=get_model('vgg_19'), layers=get_layers('vgg_19'))
model_registry['resnet101'] = ModelCommitment(identifier='resnet101', activations_model=get_model('resnet101'), layers=get_layers('resnet101'))
