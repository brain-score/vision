from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CLIP_ViT-B_32'] = ModelCommitment(identifier='CLIP_ViT-B_32', activations_model=get_model('CLIP_ViT-B_32'), layers=get_layers('CLIP_ViT-B_32'))
