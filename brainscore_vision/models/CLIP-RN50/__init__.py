from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['CLIP-RN50'] = \
    lambda: ModelCommitment(identifier='CLIP-RN50', activations_model=get_model('CLIP-RN50'), layers=get_layers('CLIP-RN50'))