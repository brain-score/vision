from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT-Small-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V2-LAST-VERSION'] = ModelCommitment(identifier='ViT-Small-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V2-LAST-VERSION', activations_model=get_model('ViT-Small-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V2-LAST-VERSION'), layers=get_layers('ViT-Small-Patch-32-224-FAT-ROTINV-INPUT-SIZE-256-CROP-SIZE-224-V2-LAST-VERSION'))
