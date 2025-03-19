from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['dinov2_base'] = lambda: ModelCommitment(identifier='dinov2_base',
                                                               activations_model=get_model('dinov2_base'),
                                                               layers=get_layers('dinov2_base'))