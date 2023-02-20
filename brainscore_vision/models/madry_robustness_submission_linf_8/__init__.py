from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['advmad_linf_8'] = ModelCommitment(identifier='advmad_linf_8', activations_model=get_model('advmad_linf_8'), layers=get_layers('advmad_linf_8'))
