from brainscore_vision import model_registry
from .model import get_model

model_registry['qwen2.5-vl-3b'] = lambda: get_model('qwen2.5-vl-3b')
