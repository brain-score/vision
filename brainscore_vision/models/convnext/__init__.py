from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['convnext_tiny'] = lambda: ModelCommitment(
    identifier='convnext_tiny', activations_model=get_model("convnext_tiny"), layers=get_layers("convnext_tiny"))

model_registry['convnext_small'] = lambda: ModelCommitment(
    identifier='convnext_small', activations_model=get_model("convnext_small"), layers=get_layers("convnext_small"))

model_registry['convnext_base'] = lambda: ModelCommitment(
    identifier='convnext_base', activations_model=get_model("convnext_base"), layers=get_layers("convnext_base"))

model_registry['convnext_large'] = lambda: ModelCommitment(
    identifier='convnext_large', activations_model=get_model("convnext_large"), layers=get_layers("convnext_large"))
