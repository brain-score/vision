from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['convnext_base_in22ft1k_256x224'] = lambda: ModelCommitment(
    identifier='convnext_base_in22ft1k_256x224',
    activations_model=get_model(),
    layers=LAYERS)
