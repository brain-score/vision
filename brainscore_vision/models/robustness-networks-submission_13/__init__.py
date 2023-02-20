from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vision_transformer_vit_large_patch16_224'] = ModelCommitment(identifier='vision_transformer_vit_large_patch16_224', activations_model=get_model('vision_transformer_vit_large_patch16_224'), layers=get_layers('vision_transformer_vit_large_patch16_224'))
