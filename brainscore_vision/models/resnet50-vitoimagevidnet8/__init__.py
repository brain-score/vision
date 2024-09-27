from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet50-vitoimagevidnet8'] = \
    lambda: ModelCommitment(identifier='resnet50-vitoimagevidnet8', activations_model=get_model('resnet50-vitoimagevidnet8'), layers=get_layers('resnet50-vitoimagevidnet8'))




