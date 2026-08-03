from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['alexnet_gated_submission2'] = lambda: ModelCommitment(
    identifier='alexnet_gated_submission2',
    activations_model=get_model(),
    layers=LAYERS)
