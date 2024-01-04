from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_base_in22ft1k_256x224'] = ModelCommitment(identifier='convnext_base_in22ft1k_256x224', activations_model=get_model('convnext_base_in22ft1k_256x224'), layers=get_layers('convnext_base_in22ft1k_256x224'))
