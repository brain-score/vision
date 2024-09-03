from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['focalnet_tiny_lrf_in1k'] = lambda: ModelCommitment(identifier='focalnet_tiny_lrf_in1k', activations_model=get_model('focalnet_tiny_lrf_in1k'), layers=get_layers('focalnet_tiny_lrf_in1k'))
