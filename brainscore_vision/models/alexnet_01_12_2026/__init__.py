from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['alexnet_01_12_2026'] = lambda: ModelCommitment(
    identifier='alexnet_01_12_2026',
    activations_model=get_model(),
    layers=LAYERS)
