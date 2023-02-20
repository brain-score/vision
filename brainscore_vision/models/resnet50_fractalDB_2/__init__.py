from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_FractalDB'] = ModelCommitment(identifier='resnet50_FractalDB', activations_model=get_model('resnet50_FractalDB'), layers=get_layers('resnet50_FractalDB'))
