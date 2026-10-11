from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers, REGION_LAYER_MAP

model_registry['vit_huge_clip224_laion_in1k'] = lambda: ModelCommitment(
    identifier='vit_huge_clip224_laion_in1k',
    activations_model=get_model('vit_huge_clip224_laion_in1k'),
    layers=get_layers('vit_huge_clip224_laion_in1k'),
    behavioral_readout_layer='fc_norm',
    region_layer_map=REGION_LAYER_MAP,
    visual_degrees=8)
