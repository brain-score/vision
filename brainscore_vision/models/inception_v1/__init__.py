from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['inception_v1'] = \
    lambda: ModelCommitment(identifier='inception_v1', activations_model=get_model('inception_v1'), layers=get_layers('inception_v1'))