from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_cutmix'] = ModelCommitment(identifier='grcnn_cutmix', activations_model=get_model('grcnn_cutmix'), layers=get_layers('grcnn_cutmix'))
