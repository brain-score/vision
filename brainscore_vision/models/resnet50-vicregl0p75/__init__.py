from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50-vicregl0p75'] = lambda: ModelCommitment(identifier='resnet50-vicregl0p75', activations_model=get_model('resnet50-vicregl0p75'), layers=get_layers('resnet50-vicregl0p75'))
