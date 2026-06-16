from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map omitted -> framework auto-selects the best candidate layer per region.
# Expectation (the iteration-2 hypothesis): the fixed VOneBlock wins V1 (and likely V2),
# while ConvNeXt layers keep V4/IT and the ConvNeXt head drives behavior.
model_registry['voneblock_convnext_large_clip_laion2b'] = lambda: ModelCommitment(
    identifier='voneblock_convnext_large_clip_laion2b',
    activations_model=get_model('voneblock_convnext_large_clip_laion2b'),
    layers=get_layers('voneblock_convnext_large_clip_laion2b'),
    behavioral_readout_layer='convnext.head.global_pool',
    visual_degrees=8,
)
