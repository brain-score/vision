from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT-Base-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-NOT-BEHAVIOR'] = ModelCommitment(identifier='ViT-Base-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-NOT-BEHAVIOR', activations_model=get_model('ViT-Base-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-NOT-BEHAVIOR'), layers=get_layers('ViT-Base-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-NOT-BEHAVIOR'))
