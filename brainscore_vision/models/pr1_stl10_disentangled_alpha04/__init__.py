from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_bibtex, get_layers, get_model, get_model_list

__all__ = ["get_bibtex", "get_layers", "get_model", "get_model_list"]

model_registry['pr1_stl10_disentangled_alpha04'] = lambda: ModelCommitment(
    identifier='pr1_stl10_disentangled_alpha04',
    activations_model=get_model('pr1_stl10_disentangled_alpha04'),
    layers=get_layers('pr1_stl10_disentangled_alpha04'),
    visual_degrees=8,
)
