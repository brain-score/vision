from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['chresmax_v3_cl_01_v3'] = lambda: ModelCommitment(identifier='chresmax_v3_cl_01_v3', activations_model=get_model('chresmax_v3_cl_01_v3'), layers=get_layers('chresmax_v3_cl_01_v3'))
