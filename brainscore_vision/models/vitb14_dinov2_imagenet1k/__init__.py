from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vitb14_dinov2_imagenet1k'] = lambda: ModelCommitment(identifier='vitb14_dinov2_imagenet1k', activations_model=get_model('vitb14_dinov2_imagenet1k'), layers=get_layers('vitb14_dinov2_imagenet1k'))
