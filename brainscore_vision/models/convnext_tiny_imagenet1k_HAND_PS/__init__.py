from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_tiny_imagenet1k_HAND_PS'] = lambda: ModelCommitment(
    identifier='convnext_tiny_imagenet1k_HAND_PS',
    activations_model=get_model('convnext_tiny_imagenet1k_HAND_PS'),
    layers=get_layers('convnext_tiny_imagenet1k_HAND_PS')
)
