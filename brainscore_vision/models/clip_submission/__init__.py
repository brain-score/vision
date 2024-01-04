from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['open_clip'] = ModelCommitment(identifier='open_clip', activations_model=get_model('open_clip'), layers=get_layers('open_clip'))
