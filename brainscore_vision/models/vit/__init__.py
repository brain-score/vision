from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['vit-base-patch16-224-in21k_debug0118'] = lambda: ModelCommitment(
    identifier='vit-base-patch16-224-in21k_debug0118',
    activations_model=get_model(),
    layers=LAYERS)
