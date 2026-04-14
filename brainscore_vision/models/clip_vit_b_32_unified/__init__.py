from brainscore_vision import model_registry
from .model import get_model

model_registry['clip-vit-b-32'] = lambda: get_model('clip-vit-b-32')
