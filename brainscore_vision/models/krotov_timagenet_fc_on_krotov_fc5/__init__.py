from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['krotov-timagenet_fc_on_krotov_fc5'] = ModelCommitment(identifier='krotov-timagenet_fc_on_krotov_fc5', activations_model=get_model('krotov-timagenet_fc_on_krotov_fc5'), layers=get_layers('krotov-timagenet_fc_on_krotov_fc5'))
