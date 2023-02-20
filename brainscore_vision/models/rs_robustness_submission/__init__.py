from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['rs_imagenet_1p00'] = ModelCommitment(identifier='rs_imagenet_1p00', activations_model=get_model('rs_imagenet_1p00'), layers=get_layers('rs_imagenet_1p00'))
