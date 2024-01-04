from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT-BLOCKS10NORM1'] = ModelCommitment(identifier='ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT-BLOCKS10NORM1', activations_model=get_model('ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT-BLOCKS10NORM1'), layers=get_layers('ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT-BLOCKS10NORM1'))
