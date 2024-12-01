from brainscore_vision import model_registry
from .model import get_model

# Register the Barlow Twins model with custom weights
model_registry['barlow_twins_custom'] = lambda: get_model('barlow_twins_custom')
