from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet-152_v1'] = \
    lambda: ModelCommitment(identifier='resnet-152_v1', activations_model=get_model('resnet-152_v1'), layers=get_layers('resnet-152_v1'))