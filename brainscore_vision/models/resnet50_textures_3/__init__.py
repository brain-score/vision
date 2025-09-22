
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_textures_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=3', activations_model=get_model('resnet50_textures_iteration=3'), layers=get_layers('resnet50_textures_iteration=3'))
