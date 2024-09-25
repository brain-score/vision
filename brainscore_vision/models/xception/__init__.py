from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet50-SIN_IN_IN'] = \
    lambda: ModelCommitment(identifier='resnet50-SIN_IN_IN', activations_model=get_model('resnet50-SIN_IN_IN'), layers=get_layers('resnet50-SIN_IN_IN'))