from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-trained-timagenet-FULL-SGD'] = ModelCommitment(identifier='lagani-trained-timagenet-FULL-SGD', activations_model=get_model('lagani-trained-timagenet-FULL-SGD'), layers=get_layers('lagani-trained-timagenet-FULL-SGD'))
