from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_272x240'] = lambda: ModelCommitment(identifier='effnetb1_272x240', activations_model=get_model('effnetb1_272x240'), layers=get_layers('effnetb1_272x240'))
