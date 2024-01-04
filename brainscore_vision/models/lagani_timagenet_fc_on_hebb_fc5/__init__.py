from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['lagani-timagenet_fc_on_hebb_fc5'] = ModelCommitment(identifier='lagani-timagenet_fc_on_hebb_fc5', activations_model=get_model('lagani-timagenet_fc_on_hebb_fc5'), layers=get_layers('lagani-timagenet_fc_on_hebb_fc5'))
