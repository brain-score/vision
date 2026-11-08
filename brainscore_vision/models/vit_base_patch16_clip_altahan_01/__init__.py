from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers, REGION_LAYER_MAP, BEHAVIORAL_READOUT_LAYER, VISUAL_DEGREES

model_registry['vit_base_patch16_clip_altahan_01'] = lambda: ModelCommitment(
    identifier='vit_base_patch16_clip_altahan_01',
    activations_model=get_model('vit_base_patch16_clip_altahan_01'),
    layers=get_layers('vit_base_patch16_clip_altahan_01'),
    region_layer_map=REGION_LAYER_MAP,
    behavioral_readout_layer=BEHAVIORAL_READOUT_LAYER,
    visual_degrees=VISUAL_DEGREES)
