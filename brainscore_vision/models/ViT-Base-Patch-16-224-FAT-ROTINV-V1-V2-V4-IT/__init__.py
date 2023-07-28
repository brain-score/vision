from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT-Base-Patch-16-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V1-V2-V4-IT'] = ModelCommitment(identifier='ViT-Base-Patch-16-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V1-V2-V4-IT', activations_model=get_model('ViT-Base-Patch-16-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V1-V2-V4-IT'), layers=get_layers('ViT-Base-Patch-16-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V1-V2-V4-IT'))
