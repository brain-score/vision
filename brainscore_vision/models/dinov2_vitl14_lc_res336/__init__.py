from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['dinov2_vitl14_lc_res336'] = lambda: ModelCommitment(
    identifier='dinov2_vitl14_lc_res336',
    activations_model=get_model('dinov2_vitl14_lc_res336'),
    layers=get_layers('dinov2_vitl14_lc_res336'),
    behavioral_readout_layer='linear_head',
    visual_degrees=8,
)
