from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CLIP2-ViT-B/32'] = ModelCommitment(identifier='CLIP2-ViT-B/32', activations_model=get_model('CLIP2-ViT-B/32'), layers=get_layers('CLIP2-ViT-B/32'))
