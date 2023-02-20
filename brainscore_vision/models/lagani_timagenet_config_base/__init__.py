from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-timagenet_base-all-hebb'] = ModelCommitment(identifier='lagani-timagenet_base-all-hebb', activations_model=get_model('lagani-timagenet_base-all-hebb'), layers=get_layers('lagani-timagenet_base-all-hebb'))
