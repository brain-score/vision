from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_ewcon_wzc'] = lambda: ModelCommitment(identifier='resnet50_ewcon_wzc', activations_model=get_model('resnet50_ewcon_wzc'), layers=get_layers('resnet50_ewcon_wzc'))
