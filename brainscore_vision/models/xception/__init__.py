from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['xception'] = \
    lambda: ModelCommitment(identifier='xception', activations_model=get_model('xception'), layers=get_layers('xception'))
