from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_SIN_IN_FT_IN'] = lambda: ModelCommitment(identifier='resnet_SIN_IN_FT_IN', activations_model=get_model('resnet_SIN_IN_FT_IN'), layers=get_layers('resnet_SIN_IN_FT_IN'))
