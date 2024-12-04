from brainscore_vision import model_registry
from .model import get_model

# Register the Barlow Twins model with custom weights
model_registry['artResNet18_1'] = lambda: get_model('artResNet18_1')
