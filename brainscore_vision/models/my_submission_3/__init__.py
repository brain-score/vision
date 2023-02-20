from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CLIP-ViT-B/32'] = ModelCommitment(identifier='CLIP-ViT-B/32', activations_model=get_model('CLIP-ViT-B/32'), layers=get_layers('CLIP-ViT-B/32'))
