from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['advmad_l2_3'] = ModelCommitment(identifier='advmad_l2_3', activations_model=get_model('advmad_l2_3'), layers=get_layers('advmad_l2_3'))
