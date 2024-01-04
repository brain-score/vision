from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT_B_16_imagenet1k'] = ModelCommitment(identifier='ViT_B_16_imagenet1k', activations_model=get_model('ViT_B_16_imagenet1k'), layers=get_layers('ViT_B_16_imagenet1k'))
model_registry['ViT_B_32_imagenet1k'] = ModelCommitment(identifier='ViT_B_32_imagenet1k', activations_model=get_model('ViT_B_32_imagenet1k'), layers=get_layers('ViT_B_32_imagenet1k'))
model_registry['ViT_L_16_imagenet1k'] = ModelCommitment(identifier='ViT_L_16_imagenet1k', activations_model=get_model('ViT_L_16_imagenet1k'), layers=get_layers('ViT_L_16_imagenet1k'))
model_registry['ViT_L_32_imagenet1k'] = ModelCommitment(identifier='ViT_L_32_imagenet1k', activations_model=get_model('ViT_L_32_imagenet1k'), layers=get_layers('ViT_L_32_imagenet1k'))
model_registry['ViT_B_16'] = ModelCommitment(identifier='ViT_B_16', activations_model=get_model('ViT_B_16'), layers=get_layers('ViT_B_16'))
model_registry['ViT_B_32'] = ModelCommitment(identifier='ViT_B_32', activations_model=get_model('ViT_B_32'), layers=get_layers('ViT_B_32'))
model_registry['ViT_L_32'] = ModelCommitment(identifier='ViT_L_32', activations_model=get_model('ViT_L_32'), layers=get_layers('ViT_L_32'))
