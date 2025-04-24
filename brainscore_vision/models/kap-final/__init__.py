from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['kap-final'] = lambda: ModelCommitment(identifier='kap-final', activations_model=get_model('kap-final'), layers=get_layers('kap-final'))
