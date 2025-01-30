from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['vgg_19'] = \
    lambda: ModelCommitment(identifier='vgg_19', activations_model=get_model('vgg_19'), layers=get_layers('vgg_19'))
