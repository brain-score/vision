from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-BEHAVIOR'] = ModelCommitment(identifier='ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-BEHAVIOR', activations_model=get_model('ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-BEHAVIOR'), layers=get_layers('ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-BEHAVIOR'))
