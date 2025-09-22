
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_z_axis_iteration=3'] = lambda: ModelCommitment(identifier='alexnet_z_axis_iteration=3', activations_model=get_model('alexnet_z_axis_iteration=3'), layers=get_layers('alexnet_z_axis_iteration=3'))
