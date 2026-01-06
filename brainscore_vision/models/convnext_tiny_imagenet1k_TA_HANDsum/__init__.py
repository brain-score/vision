from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_tiny_imagenet1k_TA_HANDsum'] = lambda: ModelCommitment(
    identifier='convnext_tiny_imagenet1k_TA_HANDsum',
    activations_model=get_model('convnext_tiny_imagenet1k_TA_HANDsum'),
    layers=get_layers('convnext_tiny_imagenet1k_TA_HANDsum')
)
