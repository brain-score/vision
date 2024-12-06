
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers
model_registry['alexnet_ambient_iteration=4'] = lambda: ModelCommitment(identifier='alexnet_ambient_iteration=4', activations_model=get_model(f'alexnet', f'ambient', f'4'), layers=get_layers(f'alexnet',f'ambient', f'4'))
        