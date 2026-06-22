from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map omitted -> framework auto-selects best layer per region.
# behavioral_readout_layer='fc_norm' = pooled (B,dim) vector for decoder-fit behaviors; label behavior
# (Geirhos/Baker) uses model(x)->1000 logits directly via the timm classifier head.
model_registry['eva02_large_in22k'] = lambda: ModelCommitment(
    identifier='eva02_large_in22k',
    activations_model=get_model('eva02_large_in22k'),
    layers=get_layers('eva02_large_in22k'),
    behavioral_readout_layer='fc_norm',
    visual_degrees=8,
)
