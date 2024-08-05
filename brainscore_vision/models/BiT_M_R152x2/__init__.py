from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['BiT-M-R152x2'] = \
    lambda: ModelCommitment(identifier='BiT-M-R152x2', activations_model=get_model('BiT-M-R152x2'), layers=get_layers('BiT-M-R152x2'))