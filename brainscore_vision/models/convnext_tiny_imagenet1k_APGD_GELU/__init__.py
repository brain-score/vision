from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_tiny_imagenet1k_APGD_GELU'] = lambda: ModelCommitment(
    identifier='convnext_tiny_imagenet1k_APGD_GELU',
    activations_model=get_model('convnext_tiny_imagenet1k_APGD_GELU'),
    layers=get_layers('convnext_tiny_imagenet1k_APGD_GELU')
)
