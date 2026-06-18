from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map omitted -> framework auto-selects best layer per region.
# behavioral_readout_layer='linear_head' -> the pooled 1000-logit output (clean vector for the
# probabilities/odd-one-out decoders). Label behavior (Geirhos/Baker) uses model(x)->1000 logits directly.
model_registry['dinov2_vitl14_lc'] = lambda: ModelCommitment(
    identifier='dinov2_vitl14_lc',
    activations_model=get_model('dinov2_vitl14_lc'),
    layers=get_layers('dinov2_vitl14_lc'),
    behavioral_readout_layer='linear_head',
    visual_degrees=8,
)
