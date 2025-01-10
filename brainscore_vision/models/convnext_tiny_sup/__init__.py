from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['convnext_tiny_sup'] = \
    lambda: ModelCommitment(identifier='convnext_tiny_sup', activations_model=get_model('convnext_tiny_sup'), layers=get_layers('convnext_tiny_sup'))

