from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_base_fb_in22k_ft_in1k'] = lambda: ModelCommitment(identifier='convnext_base_fb_in22k_ft_in1k', activations_model=get_model('convnext_base_fb_in22k_ft_in1k'), layers=get_layers('convnext_base_fb_in22k_ft_in1k'))
