from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_v2_text_noise'] = ModelCommitment(identifier='grcnn_v2_text_noise', activations_model=get_model('grcnn_v2_text_noise'), layers=get_layers('grcnn_v2_text_noise'))
