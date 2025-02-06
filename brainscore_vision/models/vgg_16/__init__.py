from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['vgg_16'] = \
    lambda: ModelCommitment(identifier='vgg_16', activations_model=get_model('vgg_16'), layers=get_layers('vgg_16'))
