
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry[f'vgg16_less_variation_iteration=1'] = lambda: ModelCommitment(identifier=f'vgg16_less_variation_iteration=1', activations_model=get_model(f'vgg16_less_variation_iteration=1'), layers=get_layers(f'vgg16_less_variation_iteration=1'))
