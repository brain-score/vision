from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['lagani-timagenet_untrained'] = lambda: ModelCommitment(
    identifier='lagani-timagenet_untrained',
    activations_model=get_model(),
    layers=LAYERS)
