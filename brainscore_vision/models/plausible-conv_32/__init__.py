from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-10W-two-blocks-LC_m'] = ModelCommitment(identifier='resnet-10W-two-blocks-LC_m', activations_model=get_model('resnet-10W-two-blocks-LC_m'), layers=get_layers('resnet-10W-two-blocks-LC_m'))
model_registry['resnet-10Wm-two-blocks-LC_m'] = ModelCommitment(identifier='resnet-10Wm-two-blocks-LC_m', activations_model=get_model('resnet-10Wm-two-blocks-LC_m'), layers=get_layers('resnet-10Wm-two-blocks-LC_m'))
