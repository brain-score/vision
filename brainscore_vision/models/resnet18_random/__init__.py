from brainscore_vision import model_registry
from .model import get_model

# Register the model with the identifier 'resnet18_random'
model_registry['resnet18_random'] = lambda: get_model('resnet18_random')
