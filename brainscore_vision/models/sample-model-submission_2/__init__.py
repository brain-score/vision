from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vgg16'] = ModelCommitment(identifier='vgg16', activations_model=get_model('vgg16'), layers=get_layers('vgg16'))
