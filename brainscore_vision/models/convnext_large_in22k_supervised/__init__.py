from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map omitted -> framework auto-selects best layer per region.
model_registry['convnext_large_in22k_supervised'] = lambda: ModelCommitment(
    identifier='convnext_large_in22k_supervised',
    activations_model=get_model('convnext_large_in22k_supervised'),
    layers=get_layers('convnext_large_in22k_supervised'),
    behavioral_readout_layer='head.global_pool',
    visual_degrees=8,
)
