from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['ViT_L_32_imagenet1k'] = lambda: ModelCommitment(identifier='ViT_L_32_imagenet1k',
                                                               activations_model=get_model('ViT_L_32_imagenet1k'),
                                                               layers=get_layers('ViT_L_32_imagenet1k'))