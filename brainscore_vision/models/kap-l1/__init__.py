from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from model import get_model, get_layers

model_registry['kap-l1'] = lambda: ModelCommitment(identifier='kap-l1', activations_model=get_model('kap-l1'), layers=get_layers('kap-l1'))
