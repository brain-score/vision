
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry[f'resnet50_less_variation_iteration=1'] = lambda: ModelCommitment(identifier=f'resnet50_less_variation_iteration=1', activations_model=get_model(f'resnet50_less_variation_iteration=1'), layers=get_layers(f'resnet50_less_variation_iteration=1'))
