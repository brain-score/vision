from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map omitted -> framework auto-selects best layer per region.
model_registry['convnextv2_huge_in22k'] = lambda: ModelCommitment(
    identifier='convnextv2_huge_in22k',
    activations_model=get_model('convnextv2_huge_in22k'),
    layers=get_layers('convnextv2_huge_in22k'),
    behavioral_readout_layer='head.global_pool',
    visual_degrees=8,
)
