from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['imagenet_l2_3_0'] = \
    lambda: ModelCommitment(identifier='imagenet_l2_3_0', activations_model=get_model('imagenet_l2_3_0'), layers=get_layers('imagenet_l2_3_0'))


