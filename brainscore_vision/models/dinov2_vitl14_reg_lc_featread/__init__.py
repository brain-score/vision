from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['dinov2_vitl14_reg_lc_featread'] = lambda: ModelCommitment(
    identifier='dinov2_vitl14_reg_lc_featread',
    activations_model=get_model('dinov2_vitl14_reg_lc_featread'),
    layers=get_layers('dinov2_vitl14_reg_lc_featread'),
    behavioral_readout_layer='feature',
    visual_degrees=8,
)
