from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['vgg-19'] = \
    lambda: ModelCommitment(identifier='vgg-19', activations_model=get_model('vgg-19'), layers=get_layers('vgg-19'))