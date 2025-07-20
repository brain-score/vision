from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['chresmax_v3_cl_01'] = lambda: ModelCommitment(identifier='chresmax_v3_cl_01', activations_model=get_model('chresmax_v3_cl_01'), layers=get_layers('chresmax_v3_cl_01'))
