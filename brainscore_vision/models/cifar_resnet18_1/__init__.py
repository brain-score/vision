from brainscore_vision import model_registry
from .model import get_model

# Register the Barlow Twins model with custom weights
model_registry['cifar_resnet18_1'] = lambda: get_model('cifar_resnet18_1')
