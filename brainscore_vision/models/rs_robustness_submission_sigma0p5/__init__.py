from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['rs_sigma_0p5'] = ModelCommitment(identifier='rs_sigma_0p5', activations_model=get_model('rs_sigma_0p5'), layers=get_layers('rs_sigma_0p5'))
