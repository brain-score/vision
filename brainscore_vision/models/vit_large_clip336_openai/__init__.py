from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vit_large_clip336_openai'] = lambda: ModelCommitment(
    identifier='vit_large_clip336_openai',
    activations_model=get_model('vit_large_clip336_openai'),
    layers=get_layers('vit_large_clip336_openai'),
    behavioral_readout_layer='fc_norm',
    visual_degrees=8)
