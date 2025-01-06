
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_wo_shading_iteration=4'] = lambda: ModelCommitment(identifier='alexnet_wo_shading_iteration=4', activations_model=get_model('alexnet_wo_shading_iteration=4'), layers=get_layers('alexnet_wo_shading_iteration=4'))
