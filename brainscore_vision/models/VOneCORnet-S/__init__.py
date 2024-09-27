from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['VOneCORnet-S'] = \
    lambda: ModelCommitment(identifier='VOneCORnet-S', activations_model=get_model('VOneCORnet-S'), layers=get_layers('VOneCORnet-S'))