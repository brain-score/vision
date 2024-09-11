from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vgg-16'] = lambda: ModelCommitment(identifier='vgg-16', activations_model=get_model('vgg-16'), layers=get_layers('vgg-16'))
