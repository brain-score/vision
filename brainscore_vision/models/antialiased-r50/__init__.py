from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['antialiased-r50'] = \
    lambda: ModelCommitment(identifier='antialiased-r50', activations_model=get_model('antialiased-r50'), layers=get_layers('antialiased-r50'))
