from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vit_huge_clip336_laion'] = lambda: ModelCommitment(
    identifier='vit_huge_clip336_laion',
    activations_model=get_model('vit_huge_clip336_laion'),
    layers=get_layers('vit_huge_clip336_laion'),
    behavioral_readout_layer='fc_norm',
    visual_degrees=8)
