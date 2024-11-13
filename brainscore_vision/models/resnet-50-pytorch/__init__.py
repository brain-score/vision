from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet-50-pytorch'] = \
    lambda: ModelCommitment(identifier='resnet-50-pytorch', activations_model=get_model('resnet-50-pytorch'), layers=get_layers('resnet-50-pytorch'))