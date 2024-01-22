from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['lagani-timagenet_base-all-hebb'] = lambda: ModelCommitment(
    identifier='lagani-timagenet_base-all-hebb',
    activations_model=get_model(),
    layers=LAYERS)
