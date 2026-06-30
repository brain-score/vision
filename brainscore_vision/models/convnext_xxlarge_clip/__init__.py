from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_xxlarge_clip'] = lambda: ModelCommitment(
    identifier='convnext_xxlarge_clip',
    activations_model=get_model('convnext_xxlarge_clip'),
    layers=get_layers('convnext_xxlarge_clip'),
    behavioral_readout_layer='head.global_pool',
    visual_degrees=8)
