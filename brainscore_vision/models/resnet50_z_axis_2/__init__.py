
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_z_axis_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=2', activations_model=get_model('resnet50_z_axis_iteration=2'), layers=get_layers('resnet50_z_axis_iteration=2'))
