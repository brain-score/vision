from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_bibtex, get_layers, get_model, get_model_list

__all__ = ["get_bibtex", "get_layers", "get_model", "get_model_list"]

model_registry['20260408_stl10_combined_alpha04'] = lambda: ModelCommitment(
    identifier='20260408_stl10_combined_alpha04',
    activations_model=get_model('20260408_stl10_combined_alpha04'),
    layers=get_layers('20260408_stl10_combined_alpha04'),
    visual_degrees=8,
)
